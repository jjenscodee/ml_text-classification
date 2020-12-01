from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW
from tqdm import tqdm, trange
import io
import numpy as np
from helpers import *
from numpy import savetxt
from numpy import asarray
import time

def train_xlnet(x_train, y_train, batch_size, lr, epochs, ids, x_test):

    # Check if there's a GPU available
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(1))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Load XLNet tokenizer
    print('Loading XLNet-base-cased tokenizer...')
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    # Set the maximum sequence length    
    MAX_LEN = 128

    print('Tokenize sentence...')
    # tokenize sentence before training
    input_ids = []
    attention_masks = []
  
    for sentence in x_train:
        encoding = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=MAX_LEN,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                )
        
        #print(len(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])))      
        print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    # convert data into torch tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    y_train = torch.tensor(y_train)

    # split our data into train and validation sets
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, y_train, 
                                                                random_state=2020, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                random_state=2020, test_size=0.1)

    # create dataloader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
    model.cuda()

    # adjust model parameter here
    param_optimizer = list(model.named_parameters())
    #no_decay = ['bias', 'gamma', 'beta']
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=lr)
    # Create the learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    # record validation accuracy per epoch
    eval_total_accuracy = []

    start = time.time()
    # start training
    for _ in trange(epochs, desc="Epoch"):
     
        # set model to training mode
        model.train()
        
        # tracking variables
        train_loss = 0
        train_steps = 0
        t0 = time.time()
        
        for step, batch in enumerate(train_dataloader):
            # update every 40 batches.
            if step % 40 == 0:
                # calculate time
                training_time = calculate_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Training time: {:}.'.format(step, len(train_dataloader), training_time))
            
            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # unpack the inputs from dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # clear out the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
                
            # backward pass
            loss.backward()
            # Clip the norm of the gradients to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # update parameters and take a step using the computed gradient
            optimizer.step()
            # Update the learning rate
            scheduler.step()
            # update tracking variables
            train_loss += loss.item()
            train_steps += 1

        print("Train loss: {}".format(train_loss/train_steps))
        
        # validation step
        # put model in evaluation mode
        model.eval()

        # tracking variables 
        eval_loss, eval_accuracy = 0, 0
        eval_steps = 0
        count = 0

        for batch in validation_dataloader:
            # add batch to GPU
            if count % 40==0:
                print("current: ", count, "total: ", len(validation_dataloader))
            count = count+1

            batch = tuple(t.to(device) for t in batch)
            # unpack the inputs from dataloader
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                # forward pass
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = output[0]
        
            # move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # update variables
            tmp_accuracy = batch_accuracy(logits, label_ids)
            eval_accuracy += tmp_accuracy
            eval_steps += 1

        eval_total_accuracy.append(eval_accuracy/eval_steps)
        print("Validation Accuracy: {}".format(eval_accuracy/eval_steps))

    end = time.time()
    total_time = calculate_time(end-start)

    # save tracking variables
    tracking = eval_total_accuracy
    tracking.append(total_time)
    print(tracking)
    savetxt('tracking_xlnet.txt', tracking, fmt='%s')

    print("Training complete")

    print("Start testing")

    input_ids = []
    attention_masks = []

    # create token
    for sentence in x_test:
        encoding = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=MAX_LEN,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                )

        print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    prediction_inputs = torch.cat(input_ids, dim=0)
    prediction_masks = torch.cat(attention_masks, dim=0)

    # create dataloader
    prediction_data = TensorDataset(prediction_inputs, prediction_masks,)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # put model in evaluation mode
    model.eval()

    # tracking variables 
    predictions = []

    for batch in prediction_dataloader:
        # add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # unpack the inputs from dataloader
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            # forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        # move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        
        # store predictions
        predictions.append(logits)

    # combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    print(flat_predictions)
    print('DONE.')

    return flat_predictions
