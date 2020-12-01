import numpy as np
from helpers import *
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import torch
from numpy import savetxt
from numpy import asarray

def train_bert(x_train, y_train, batch_size, lr, epochs, ids, x_test):

    # Check if there's a GPU available
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(1))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # load BERT tokenizer
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    print('Tokenize sentence...')
    # tokenize sentence before training
    input_ids = []

    for sentence in x_train:
        encoding = tokenizer.encode(
                            sentence,
                            add_special_tokens = True)
        
        # add the encoded sentence to the list
        input_ids.append(encoding)
    
    # Set the maximum sequence length
    MAX_LEN = 128
    # add pad at the end of sentence
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                            value=0, truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    for sentence in input_ids:
        # if token id=0 indicate padding -> set mask to 0
        tmp_mask = [int(token_id > 0) for token_id in sentence]
        # Store the attention mask for this sentence.
        attention_masks.append(tmp_mask)

    # split our data into train and validation sets
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, y_train, 
                                                                random_state=2020, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, y_train,
                                                random_state=2020, test_size=0.1)

    # convert data into torch tensors
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # create dataloader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top 
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels = 2,  
        output_attentions = False, 
        output_hidden_states = False, 
    )
    model.cuda()

    # AdamW is a class from the huggingface library
    optimizer = AdamW(model.parameters(),lr = lr,eps = 1e-8 )

    # Create the learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    # record validation accuracy per epoch
    eval_total_accuracy = []

    start = time.time()
    # start training
    for epoch_i in range(0, epochs):

        # set model to training mode
        model.train()
        
        # tracking variables
        train_loss = 0
        t0 = time.time()

        for step, batch in enumerate(train_dataloader):
            # update every 40 batches
            if step % 40 == 0:
                # calculate time
                training_time = calculate_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), training_time))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # clear out the gradients
            model.zero_grad()        
            # forward pass
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

            # update tracking variables
            loss = outputs[0]
            train_loss += loss.item()

            # backward pass
            loss.backward()
            # Clip the norm of the gradients to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = train_loss / len(train_dataloader)            

        print("average training loss: {}".format(avg_train_loss))

        #validation step
        print("Running Validation...")
        
        # put model in evaluation mode
        model.eval()
        # Tracking variables 
        t0 = time.time()
        eval_loss, eval_accuracy = 0, 0
        eval_steps = 0
        count = 0
        
        for batch in validation_dataloader:
            if count % 40==0:
                print("current: ", count, "total: ", len(validation_dataloader))
            count = count+1            
            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            
            # unpack the inputs from dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():    
                # forward pass    
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            
            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # update variables
            tmp_accuracy = batch_accuracy(logits, label_ids)
            eval_accuracy += tmp_accuracy
            eval_steps += 1

        eval_total_accuracy.append(eval_accuracy/eval_steps)
        print("Validation  Accuracy: {}".format(eval_accuracy/eval_steps))

    end = time.time()
    total_time = calculate_time(end-start)

    # save tracking variables
    tracking = eval_total_accuracy
    tracking.append(total_time)
    savetxt('tracking_bert.txt', tracking, fmt='%s')

    print("Training complete!")

    print("Start testing")

    # tokenize sentence
    input_ids = []

    for sentence in x_test:
        encoded_sent = tokenizer.encode(
                            sentence,                     
                            add_special_tokens = True, 
                    )
        
        input_ids.append(encoded_sent)

    # Pad tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                            dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 

    # Convert to tensors
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
 
    # Create DataLoader
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    # put model in evaluation mode
    model.eval()
    # Tracking variables 
    predictions = []
    
    for batch in prediction_dataloader:

        # add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # unpack the inputs from dataloader
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]
        # Move logits to CPU
        logits = logits.detach().cpu().numpy()
    
        # Store predictions
        predictions.append(logits)

    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    print(flat_predictions)
    print('DONE.')

    return flat_predictions
