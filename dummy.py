class GroundTruthData(object):
    """Abstract class for data sets that are two-step generative models."""
    @property
    def num_factors(self):
        raise NotImplementedError()

    @property
    def factors_num_values(self):
        raise NotImplementedError()

    @property
    def observation_shape(self):
        raise NotImplementedError()

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]


class DummyData(GroundTruthData):
    """Dummy image data set of random noise used for testing."""

    @property
    def num_factors(self):
        return 10

    @property
    def factors_num_values(self):
        return [5] * 10

    @property
    def observation_shape(self):
        return [64, 64, 1]

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return random_state.randint(5, size=(num, self.num_factors))

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        return random_state.random_sample(size=(factors.shape[0], 64, 64, 1))