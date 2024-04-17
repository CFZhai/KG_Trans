# KG_Trans
This is an initial example demonstration of reproducibility for the KDD 2024 working paper titled "Enhancing Knowledge Retention: A Fusion Transformer Model with Knowledge Graph Embeddings."

## Brief introduction of the model 
Knowledge learning and retention are perennial challenges for human beings. In this study, we develop a machine learning model
to predict the success rate of learning each knowledge entity in an
upcoming session based on the user’s learning history. We begin by
learning the latent representation of each knowledge entity within
a knowledge graph. This graph captures both the inherent intellectual relationships between different knowledge entities and the
learning patterns of individuals across these entities. Leveraging
this informative knowledge graph, we extract the latent representation of each knowledge entity, which is then used to represent
the user’s learning history. Employing the Transformer model to
process the user’s learning history, our goal is to predict the success
rate of learning each knowledge entity in the upcoming session.
This integration of two modules enables us to forecast learner recall probabilities with unprecedented precision. By evaluating the
model on 23 million study logs from an English vocabulary learning platform, we demonstrate that our proposed model surpasses
state-of-the-art benchmarks, achieving a 13.14% increase in knowledge retention prediction accuracy and a 33.3% rise in the AUC
metric. These significant improvements not only establish a new
benchmark in the field but also pave the way for enhancing digital
learning platforms, promising a more personalized, efficient, and
effective learning process.

![Model Structure](./KG_Transl_structure.png)

