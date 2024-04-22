class DataSplit:
	def __init__(self, **kwargs):
		self._training_ratio = kwargs.get('training_split_ratio', 0.8)
		self._test_ratio = kwargs.get('testing_split_ratio', 0.1)
		self._validation_ratio = kwargs.get('validation_split_ratio', 0.1)
		assert self._training_ratio >= 0 and self._test_ratio >= 0 and self._validation_ratio >= 0, "The ratios must be positive"
		assert round(self.training_ratio + self.testing_ratio + self.validation_ratio, 5) == 1.0, "The sum of the ratios must be 1.0"
	
	@property
	def training_ratio(self):
		return self._training_ratio
	
	@property
	def testing_ratio(self):
		return self._test_ratio
	
	@property
	def validation_ratio(self):
		return self._validation_ratio