def add_histogram_embeddings(self, source_batch, target_batch):
        source_embeddings = self.backbone(source_batch)
        target_embeddings = self.backbone(target_batch)
        self.logger.experiment.add_histogram("Source_Embedding", source_embeddings, self.current_epoch)
        self.logger.experiment.add_histogram("Target_Embedding", target_embeddings, self.current_epoch)
        
def add_histogram_parameters(self):
    # iterating through all parameters
    for name, params in self.named_parameters():
        self.logger.experiment.add_histogram(name.replace(".", "/"),params,self.current_epoch)