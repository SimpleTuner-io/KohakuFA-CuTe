"""Apply unchanged L40S arithmetic gates to the proposed causal CTA groups."""

from benchmarks import validate_l40s_static as validator
from benchmarks.l40s_causal_groups import make_direct


def main():
    validator.make_attention = make_direct
    validator.main()


if __name__ == "__main__":
    main()
