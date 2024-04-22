class DatasetSettings:
	
	def __init__(self, minimum_number_of_items_for_class:int):
		self._minimum_number_of_items_for_class = minimum_number_of_items_for_class
		
	@property
	def minimum_number_of_items_for_class(self):
		return self._minimum_number_of_items_for_class