import torch
import numpy as np

class ProjectorTensorBoard:
    
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.source_embeddings = []
        self.source_embeddings_labels = []
        self.target_embeddings = []
        self.target_embeddings_labels = []
        self.source_images = []
        self.target_images = []
        
    def add_source_embeddings(self, source_embeddings, labels_embeddings, images):
        self.source_embeddings.extend(source_embeddings)
        self.source_embeddings_labels.extend(labels_embeddings)
        self.source_images.extend(images)
    
    def add_target_embeddings(self, target_embeddings, labels_embeddings, images):
        self.target_embeddings.extend(target_embeddings)
        self.target_embeddings_labels.extend(labels_embeddings)
        self.target_images.extend(images)
                  
    def log_embeddings(self, name_labels, num_samples):
        self.source_embeddings = torch.stack(self.source_embeddings)
        self.target_embeddings = torch.stack(self.target_embeddings)

        self.source_embeddings =  self.source_embeddings[0:num_samples, :]
        self.source_embeddings_labels = torch.stack(self.source_embeddings_labels)
        self.source_embeddings_labels =  self.source_embeddings_labels[0:num_samples]
        self.source_images = torch.stack(self.source_images)
        self.source_images = self.source_images[0:num_samples, :]
        
        self.target_embeddings = self.target_embeddings[0:num_samples, :]
        self.target_embeddings_labels = torch.stack(self.target_embeddings_labels)
        self.target_embeddings_labels =  self.target_embeddings_labels[0:num_samples]
        self.target_images = torch.stack(self.target_images)
        self.target_images = self.target_images[0:num_samples, :]
        
        embeddings = torch.cat((self.source_embeddings, self.target_embeddings), 0)
        images = torch.cat(( self.source_images, self.target_images), 0)
                
        #Source
        domain_source = ["source" for i in range(self.source_embeddings.shape[0])]
        self.source_embeddings_labels = [name_labels[idx_labels] for idx_labels in self.source_embeddings_labels]
        source_metadata =  [ (domain_source[idx], self.source_embeddings_labels[idx]) for idx in range(len(domain_source))]
        
        #Target
        domain_target = np.array(["target" for i in range(self.target_embeddings.shape[0])])
        self.target_embeddings_labels = [name_labels[idx_labels] for idx_labels in self.target_embeddings_labels]
        target_metadata =  [ (domain_target[idx], self.target_embeddings_labels[idx]) for idx in range(len(domain_target))]
        
        source_metadata.extend(target_metadata)
        
        self.logger.experiment.add_embedding(embeddings, metadata=source_metadata, label_img=images, metadata_header=['Domains', 'Labels']) 