from src.dataset.settings.dataset_settings import DatasetSettings


class DatasetGenerator:
	def __init__(self, dataset_settings: DatasetSettings):
		self.dataset_settings = dataset_settings

	def process_ifc_entities(self):
		pass
	
	def create_datasets(self):
		pass

	@property
	def dataset_settings(self):
		return self._dataset_settings
