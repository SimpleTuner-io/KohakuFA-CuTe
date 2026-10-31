"""Precision policy shared by the qualified warp-MMA device schedules."""


def schedule(program, dimension):
    from .weight_precision import schedule as generic_schedule

    return generic_schedule(program, dimension)
