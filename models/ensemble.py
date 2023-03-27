import torch


class ModelsEnsemble():
    """Ensemble of models
    Predicts the class with the highest average probability

    Requires models to have predict_proba method.
    """
    def __init__(self, models):
        self.models = models
    
    def predict_proba(self, dataloader, device):
        predictions = torch.zeros((len(dataloader.dataset), 10))
        batch_size = next(iter(dataloader))[0].shape[0]

        device = torch.device(device)
        for model in self.models:
            model.to(device)
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model.predict_proba(images).cpu()
                    predictions[i*batch_size:(i+1)*batch_size] += outputs
                predictions = predictions / len(self.models)
        return predictions
    
    def predict(self, dataloader, device):
        predictions = self.predict_proba(dataloader, device)
        prediction_probability, prediction_class_id = torch.max(predictions, dim=1)
        return prediction_class_id

