# Contrastive Learning via Siamese Neural Networks for conditional average treatment effects (CATE) estimation

> Codice a supporto della tesi di laurea magistrale: **“Contrastive Learning via Siamese Neural Networks for Conditional Average Treatment Effect (CATE) Estimation”**.  
> Estende un modello in stile **BCAUSS** con una regolarizzazione **siamese/contrastive** che struttura lo spazio latente: avvicina unità con **ITE simile** e allontana unità con **ITE diverso**.

## Struttura del repository

<code>
├── outputs_jobs/ # Output degli esperimenti su JOBS
├── results/ # Risultati aggregati (CSV, tabelle)
├── saved_weights_reps/ # Pesi pre-addestrati per trial/replica
├── src/
│ ├── bcauss/ # Componenti del modello base
│ ├── data/ # (Metti qui i dataset se non gestiti dal loader)
│ ├── dataset_jobs/ # Utility per il benchmark JOBS
│ ├── evaluation_results/ # Metriche & plot dalle run di valutazione
│ ├── evaluation_results_refactored/# Esportazioni “pulite” più recenti
│ ├── extraction_results/ # Embedding & trattamenti salvati
│ ├── models/ # Wrapper del modello e teste (h0, h1, ...)
│ ├── outputs/ # Artefatti vari delle run
│ ├── siamese_bcuass/ # Training siamese/contrastive (BCAUSS)
│ ├── tests/
│ │ ├── init.py
│ │ ├── advantages.py
│ │ ├── contrastive.py
│ │ ├── data_loader.py
│ │ └── metrics.py
│ └── evaluation.py # Valutazione completa su IHDP (per replica)
└── README.md
</code>



> Nota: gli import usano il pacchetto `src/`. Se serve, aggiungi la root del repo al `PYTHONPATH`.
