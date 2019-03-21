# ProgettoDIA 2019

alessandro.nuara@polimi.it

Pricing project about Samsung Galaxy S10




1. Choose a product sold on the Internet. Provide a brief description.

2. Imagine an aggregate demand curve aggregating three different demand curves. Provide a description of the three classes of users corresponding to the three disaggregate demand curve. For every disaggregate demand curve, define the phases, providing practical motivations based on market evidence (e.g., seasons), and, for every phase, define the probability distribution, even subject to smooth changes. Note: the definition of the classes of the users must be done by introducing features and different values for the features.

3. Define the horizon for the optimization based on market evidence and choose a number of discrete values of the price accordingly.

4. Apply to the aggregate demand curve the following algorithms and show, in a plot, how the regret and the reward vary in time:
    - k-testing (choose at beginning the number of samples of the experiment and then apply the hypothesis test);
    - UCB1, TS;
    - SW-UCB1, SW-TS (motivate the length of the sliding window).

5. Suppose to apply, the first day of every week, an algorithm to identify contexts and, therefore, to disaggregate the demand curves if doing that is the best we can do. And, if such an algorithm suggests disaggregating the demand curve at time t, then, from t on, keep such demand curves disaggregate. In order to disaggregate the demand curve, it is necessary to reason on the features and the values of the features. Apply the following algorithms and show, in a plot, how the regret and the reward vary in time (also comparing the regret and the reward of these algorithms with those obtained when the algorithms are applied to the aggregate demand curve):
    - UCB1, TS;
    - SW-UCB1, SW-TS (motivate the length of the sliding window).
