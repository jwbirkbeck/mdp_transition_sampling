import numpy as np
from src.utils.consts import task_pool as global_task_pool

class MWNSDistribution:
    def __init__(self, seed, state_space, action_space, current_task):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        subseeds = self.rng.choice(int(1e6), size=4, replace=False)
        self.sensor_dist = SensActNSDistribution(rng=np.random.default_rng(subseeds[0]), space=state_space)
        self.actuator_dist = SensActNSDistribution(rng=np.random.default_rng(subseeds[1]), space=action_space)
        self.maint_dist = MaintenanceNSDist(rng=np.random.default_rng(subseeds[2]))
        self.task_dist = TaskNSDist(rng=np.random.default_rng(subseeds[3]), current_task=current_task)

    def generate_sequence(self, sequence_length, append=False):
        self.sequence_length = sequence_length
        self.maint_dist.generate_sequence(sequence_length=sequence_length)
        self.task_dist.generate_sequence(sequence_length=sequence_length)
        self.sensor_dist.generate_sequence(sequence_length=sequence_length, maintenance=self.maint_dist.seq, append=append)
        self.actuator_dist.generate_sequence(sequence_length=sequence_length, maintenance=self.maint_dist.seq, append=append)

    def step_with_ns(self, action, env):
        # Apply actuator to action transform, then step, then apply sensor transform to next_state
        processed_action = self.actuator_dist.apply_ns(raw_input=action)
        state, reward, done, truncated, info = env.env.step(processed_action)
        processed_state = self.sensor_dist.apply_ns(raw_input=state)
        self.task_dist.apply_ns(env=env)
        return processed_state, reward, done, truncated, info

    def freeze(self):
        self.task_dist.frozen = True
        self.sensor_dist.frozen = True
        self.actuator_dist.frozen = True

    def unfreeze(self):
        self.task_dist.frozen = False
        self.sensor_dist.frozen = False
        self.actuator_dist.frozen = False

    def set_sequence_ind(self, ind):
        self.task_dist.seq_ind = ind
        self.sensor_dist.seq_ind = ind
        self.actuator_dist.seq_ind = ind

    def restart_sequence(self):
        self.maint_dist.seq_ind = 0
        self.task_dist.seq_ind = 0
        self.sensor_dist.seq_ind = 0
        self.actuator_dist.seq_ind = 0


class SensActNSDistribution:
    def __init__(self, rng, space):
        self.rng = rng
        self.space = space
        self.dim = len(self.space.high)
        self.prev_output = np.zeros(shape=self.dim)

        self.set_degradation_params()
        self.set_failure_params()
        self.seq_ind = None
        self.bias_seq = None
        self.bias_direction = None
        self.error_seq = None
        self.error_param_seq = None
        self.failure_seq = None

        self.frozen = False

    def set_failure_params(self, p_failure=1e-6, p_inplace=0.45, p_zero=0.45, p_max=0.05, p_min=0.05):
        total_p_failure = p_failure / self.dim
        self.failure_params = {'p_inplace': total_p_failure * p_inplace,
                               'p_zero': total_p_failure * p_zero,
                               'p_max': total_p_failure * p_max,
                               'p_min': total_p_failure * p_min}

    def set_degradation_params(self, bias_prob=1e-3, error_prob=1e-3, bias_max_delta=0.01, error_max_delta=0.01):
        bias_prob /= self.dim
        error_prob /= self.dim
        self.degrad_params = {'p_bias': bias_prob,
                              'p_error': error_prob,
                              'bias_max_delta': bias_max_delta,
                              'error_max_delta': error_max_delta}

    def generate_sequence(self, sequence_length, maintenance, append=False):
        if append:
            last_bias = self.bias_seq[-1, :].reshape(1, -1)
            last_error_params = self.error_param_seq[-1, :].reshape(1, -1)
            last_failures = self.failure_seq[-1, :].reshape(1, -1)
        else:
            last_bias = np.zeros(shape=[1, self.dim])
            last_error_params = np.zeros(shape=[1, self.dim])
            last_failures = np.zeros(shape=[1, self.dim])

        if self.degrad_params['p_bias'] == 0.0:
            biases = np.repeat(last_bias, repeats=sequence_length)
        else:
            p_bias = [1 - self.degrad_params['p_bias'], self.degrad_params['p_bias']]
            biases = np.zeros(shape=[sequence_length, self.dim])
            prev_ind = 0
            for ind in list(np.where(maintenance == 1)[0]) + [sequence_length]:
                split_length = ind - prev_ind
                bias_events = self.rng.choice(a=np.array([0, 1]), size=[split_length, self.dim], p=p_bias)
                if self.bias_direction is None:
                    self.bias_direction = self.rng.choice(a=np.array([-1, 1]), size=[1, self.dim], p=[0.5, 0.5])
                bias_magnitudes = self.rng.uniform(low=0, high=self.degrad_params['bias_max_delta'],
                                                   size=[split_length, self.dim])
                if prev_ind != 0:
                    biases[prev_ind:ind, :] = np.cumsum(bias_events * self.bias_direction * bias_magnitudes, axis=0)
                else:
                    appended_bias_calcs = np.append(last_bias, bias_events * self.bias_direction * bias_magnitudes, axis=0)
                    patched_cumsum = np.cumsum(appended_bias_calcs, axis=0)
                    patched_cumsum = patched_cumsum[1:, :]
                    biases[prev_ind:ind, :] = patched_cumsum
                prev_ind = ind
            del p_bias, prev_ind, split_length, appended_bias_calcs, patched_cumsum, bias_events, bias_magnitudes

        if self.degrad_params['p_error'] == 0.0:
            error_params = np.repeat(last_error_params, repeats=sequence_length)
        else:
            p_error = [1 - self.degrad_params['p_error'], self.degrad_params['p_error']]
            error_params = np.zeros(shape=[sequence_length, self.dim])
            prev_ind = 0
            for ind in list(np.where(maintenance == 1)[0]) + [sequence_length]:
                split_length = ind - prev_ind
                error_events = self.rng.choice(a=np.array([0, 1]), size=[split_length, self.dim], p=p_error)
                bias_magnitudes = self.rng.uniform(low=0, high=self.degrad_params['error_max_delta'],
                                                   size=[split_length, self.dim])
                if prev_ind != 0:
                    error_params[prev_ind:ind, :] = np.cumsum(error_events * bias_magnitudes, axis=0)
                else:
                    appended_error_calcs = np.append(last_error_params, error_events * bias_magnitudes, axis=0)
                    patched_cumsum = np.cumsum(appended_error_calcs, axis=0)
                    patched_cumsum = patched_cumsum[1:, :]
                    error_params[prev_ind:ind, :] = patched_cumsum
                prev_ind = ind
            del p_error, prev_ind, split_length, appended_error_calcs, patched_cumsum, error_events, bias_magnitudes
            self.error_param_seq = error_params
            errors = self.rng.normal(loc=0, scale=self.error_param_seq)

        # Failure events:
        if sum(self.failure_params.values()) == 0.0:
            failures = np.repeat(last_failures, repeats=sequence_length)
        else:
            p_options = range(len(self.failure_params.keys()) + 1)
            p_failure = [1 - sum(self.failure_params.values())] + list(self.failure_params.values())
            failures = self.rng.choice(a=p_options, size=[sequence_length, self.dim], p=p_failure)
            prev_ind = 0
            for ind in list(np.where(maintenance == 1)[0]) + [int(sequence_length)]:
                if prev_ind == 0:
                    this_split = np.append(last_failures, failures[prev_ind:ind, :], axis=0)
                else:
                    this_split = failures[prev_ind:ind, :]
                first_failures = (this_split != 0).argmax(axis=0)
                col_range = range(failures.shape[1])
                for first_failure, col in zip(first_failures, col_range):
                    this_split[first_failure:, col] = this_split[first_failure, col]
                prev_ind = ind
        del p_options, p_failure, prev_ind, this_split, first_failures

        self.bias_seq = biases
        self.error_seq = errors
        self.failure_seq = failures
        self.seq_ind = 0

    def apply_ns(self, raw_input):
        output = raw_input.copy()
        assert len(raw_input.shape) == 1
        # apply bias and error
        bias_row = self.bias_seq[self.seq_ind]
        if not self.frozen:
            error_row = self.error_seq[self.seq_ind]
        else:
            error_row = self.rng.normal(loc=0.0, scale=self.error_param_seq[self.seq_ind])
        output += (bias_row + error_row)
        # apply failure
        failure_row = self.failure_seq[self.seq_ind, :]
        if sum(failure_row) > 0:
            for column in range(raw_input.shape[0]):
                if failure_row[column] != 0:
                    # 1: lock in place
                    if failure_row[column] == 1:
                        output[column] = self.prev_output[column]
                    # 2: lock to zero
                    elif failure_row[column] == 2:
                        output[column] = 0.0
                    # 3: lock to max
                    elif failure_row[column] == 3:
                        output[column] == self.space.high[column]
                    # 4: lock to min
                    elif failure_row[column] == 4:
                        output[column] == self.space.low[column]
        # Clamp to enforce usual space bounds for state/action:
        low = self.space.low
        high = self.space.high
        output = np.clip(a=output, a_min=low, a_max=high)
        self.prev_output = output
        if not self.frozen:
            self.seq_ind += 1
        return output


class MaintenanceNSDist:
    def __init__(self, rng):
        self.rng = rng
        self.set_prob_maintenance()

    def set_prob_maintenance(self, probability=1e-6):
        self.p_maint = probability

    def generate_sequence(self, sequence_length):
        if self.p_maint == 0.0:
            self.seq = np.zeros(shape=sequence_length)
            self.seq_ind = 0
        else:
            self.seq = self.rng.choice([0, 1], size=sequence_length, p=[1 - self.p_maint, self.p_maint])
            self.seq_ind = 0


class TaskNSDist:
    def __init__(self, rng, current_task, task_pool=None):
        self.rng = rng
        self.current_task = current_task
        self.set_prob_task_change()
        if task_pool is not None:
            self.task_pool = np.array(task_pool, dtype='object')
        else:
            self.task_pool = np.array(global_task_pool, dtype='object')
        self.task_indices = np.arange(self.task_pool.shape[0], dtype=np.int8)

        self.frozen = False

    def set_prob_task_change(self, probability=1e-5):
        self.p_task_change = probability

    def generate_sequence(self, sequence_length):
        curr_task_index = self.task_indices[self.task_pool == self.current_task].item()
        assert len(self.task_indices) < 127, "int8 indices cannot represent more than 127 tasks"
        tasks = np.full(fill_value=curr_task_index, shape=[sequence_length], dtype=np.int8)
        probs = [self.p_task_change, 1 - self.p_task_change]
        task_changes = self.rng.choice(np.array([True, False], dtype=bool), size=sequence_length, p=probs)
        prev_ind = 0
        for ind in list(np.where(task_changes)[0]) + [int(sequence_length)]:
            if prev_ind == 0:
                prev_ind = ind
                continue
            new_task = self.rng.choice(a=self.task_indices[self.task_indices != curr_task_index], size=1).item()
            tasks[prev_ind:ind] = new_task
            prev_ind = ind
            curr_task_index = new_task
        self.seq = tasks
        self.seq_ind = 0

    def apply_ns(self, env):
        if self.task_pool[self.seq[self.seq_ind]] != self.current_task:
            new_task_ind = self.seq[self.seq_ind]
            new_task = self.task_pool[new_task_ind]
            env.change_task(task_name=new_task)
        if not self.frozen:
            self.seq_ind += 1
        return env
