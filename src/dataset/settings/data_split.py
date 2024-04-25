class DataSplit:
	"""
	Class for splitting data into training, testing, and validation sets based on specified ratios.

	Attributes:
		training_ratio (float): Proportion of the data to be used for training.
		testing_ratio (float): Proportion of the data to be used for testing.
		validation_ratio (float): Proportion of the data to be used for validation.

	Raises:
		AssertionError: If any of the input ratios are negative.
		AssertionError: If the sum of all ratios does not equal 1.0.
	"""
	
	def __init__(self, **kwargs):
		"""
		Initializes the DataSplit class with the ratios for splitting data.

		Args:
			**kwargs: Arbitrary keyword arguments. Expected keys are:
				'training_split_ratio' (float): Ratio of data to be used for training (default: 0.8).
				'testing_split_ratio' (float): Ratio of data to be used for testing (default: 0.1).
				'validation_split_ratio' (float): Ratio of data to be used for validation (default: 0.1).
		"""
		self._training_ratio = kwargs.get('training_split_ratio', 0.8)
		self._test_ratio = kwargs.get('testing_split_ratio', 0.1)
		self._validation_ratio = kwargs.get('validation_split_ratio', 0.1)
		assert self._training_ratio >= 0 and self._test_ratio >= 0 and self._validation_ratio >= 0, "The ratios must be positive"
		assert round(self._training_ratio + self._test_ratio + self._validation_ratio,
					 5) == 1.0, "The sum of the ratios must be 1.0"
	
	@property
	def training_ratio(self):
		"""
		float: Gets the training data split ratio.
		"""
		return self._training_ratio
	
	@property
	def testing_ratio(self):
		"""
		float: Gets the testing data split ratio.
		"""
		return self._test_ratio
	
	@property
	def validation_ratio(self):
		"""
		float: Gets the validation data split ratio.
		"""
		return self._validation_ratio
