from src.metaworld.wrapper import MetaWorldWrapper

class WrappedButtonPressTopdown(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='button-press-topdown-v2')


class WrappedButtonPress(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='button-press-v2')


class WrappedButtonPressWall(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='button-press-wall-v2')


class WrappedCoffeeButton(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='coffee-button-v2')


class WrappedCoffeePush(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='coffee-push-v2')

class WrappedDialTurn(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='dial-turn-v2')


class WrappedDoorClose(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='door-close-v2')


class WrappedDoorUnlock(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='door-unlock-v2')


class WrappedHandlePressSide(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='handle-press-side-v2')


class WrappedHandlePress(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='handle-press-v2')


class WrappedPegInsertSide(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='peg-insert-side-v2')


class WrappedPlateSlideBack(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='plate-slide-back-v2')


class WrappedPlateSlide(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='plate-slide-v2')


class WrappedPushBack(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='push-back-v2')


class WrappedReach(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='reach-v2')


class WrappedReachWall(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='reach-wall-v2')


class WrappedSoccer(MetaWorldWrapper):
    def __init__(self):
        super().__init__(task_name='soccer-v2')
