# CATE-SNN


Il codice è attualmente in fase di validazione; potrebbero essere necessari aggiustamenti per ottenere una convergenza più rapida e stabile. Di seguito alcune indicazioni preliminari:

1. **Rimozione di BatchNorm nella representation network**  
   Nel costruttore della rete di rappresentazione è presente una linea simile a:  
   ```python
   layers.append(nn.BatchNorm1d(hidden_dim))
   ```  
   Commentare o rimuovere tale riga può agevolare la discesa della *factual loss* e della *contrastive loss*, evitando possibili effetti indesiderati del rumore di batch sulle repliche.

2. **Normalizzazione a monte dei dati**  
   Si raccomanda di applicare una standardizzazione (ad esempio `StandardScaler`) o altra forma di normalizzazione sui features **prima** dell’input in rete, piuttosto che utilizzare BatchNorm interno. In questo modo i dati in ingresso risultano più “stabili” e favoriscono una migliore ottimizzazione.

3. **Aggregazione delle loss e aggiornamento dei pesi**  
   Le loss (factual e contrastive) devono essere accumulate su tutte le repliche all’interno di un’epoca, chiamando poi:
   ```python
   total_loss.backward()
   optimizer.step()
   ```
   **una sola volta per epoca**, anziché eseguire un `backward()` e uno `step()` dopo ogni realizzazione. Ciò garantisce aggiornamenti più robusti e coerenti con l’obiettivo di minimizzare la loss aggregata.

