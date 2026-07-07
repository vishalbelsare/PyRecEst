"""Demonstrate a survival-aware CRP-style association prior.

The example contrasts an ordinary CRP-style mass with a tracking-specific
alternative in which an old, weakly visible track is less attractive for a new
measurement than a recently seen visible track, while still retaining separate
birth and clutter alternatives.
"""

from pyrecest.filters.survival_aware_crp import (
    SurvivalAwareCRPAssociationPrior,
    posterior_existence_after_missed_detection,
)


def main():
    prior = SurvivalAwareCRPAssociationPrior(
        discount=0.0,
        strength=2.0,
        time_decay=0.8,
    )

    probabilities = prior.predictive_assignment_probabilities(
        track_masses=[8.0, 8.0],
        survival_probabilities=[0.95, 0.6],
        detection_probabilities=[0.9, 0.9],
        visibility_probabilities=[1.0, 0.25],
        compatibility_scores=[0.8, 0.8],
        time_since_seen=[0.0, 4.0],
        birth_rate=0.5,
        clutter_rate=0.1,
    )

    print("P(track 0), P(track 1), P(birth), P(clutter):")
    print(tuple(round(probability, 3) for probability in probabilities))

    visible_miss = posterior_existence_after_missed_detection(0.8, 0.9, 1.0)
    occluded_miss = posterior_existence_after_missed_detection(0.8, 0.9, 0.1)

    print("existence after visible miss: ", round(visible_miss, 3))
    print("existence after occluded miss:", round(occluded_miss, 3))


if __name__ == "__main__":
    main()
