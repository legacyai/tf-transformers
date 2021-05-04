import tensorflow as tf 
import time

def trainer(model,
            batch_size,
            dataset,
            loss_fn,
            optimizer,
            epochs,
            validation_dataset,
            validation_loss_fn,
            model_checkpoint_dir,
            steps_per_epoch=100,
            model_save_interval_epochs=1,
            max_number_of_models = 10,
            total_examples=None,
            model_save_interval_steps=1,
            overwrite_checkpoint_dir=False,
            validation_interval_steps=None,
            train_steps=None,
            steps_per_call=100,
            eval_callback=None
           ):
    
    # Calculate it, better provided (avoid lag)
    if total_examples is None:
        for counter, _ in enumerate(dataset):
            pass
        total_examples = counter * batch_size

    # no of batches = steps per epoch
    if steps_per_epoch is None:
        steps_per_epoch =  (total_examples//batch_size)
        
    if not overwrite_checkpoint_dir:
        import os
        if os.path.exists(model_checkpoint_dir):
            raise FileExistsError("Model directory exists")
        
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=model_checkpoint_dir, max_to_keep=max_number_of_models)
        
    @tf.function
    def train(iterator):
        """The step function for one training step"""

        def train_step(batch_inputs, batch_labels):
            """The computation to run on each TPU device."""
            with tf.GradientTape() as tape:
                model_outputs = model(batch_inputs)
                loss = loss_fn(batch_labels, model_outputs)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            # training_loss.update_state(loss * strategy.num_replicas_in_sync)
            training_loss.update_state(loss)
            current_lr = optimizer._decayed_lr(tf.float32)
            learning_rate_holder.update_state(current_lr)

        for _ in tf.range(tf.convert_to_tensor(steps_per_call)):
            batch_inputs, batch_labels = next(iterator)
            train_step(batch_inputs, batch_labels)
            
    @tf.function
    def validate(iterator):
        """Validation step"""
        for (batch_inputs, batch_labels) in iterator:
            model_outputs = model(batch_inputs)
            loss = validation_loss_fn(batch_labels, model_outputs)
            validation_loss.update_state(loss)
            
    # do validation
    def do_validation(validation_dataset):
        validate(validation_dataset)
        loss = validation_loss.result()
        validation_loss.reset_states()
        score = None
        val_loss = None
        if eval_callback:
            score = eval_callback(kwargs)
        return {'val_loss': val_loss , 'val_score': score}
        
    # necessary metrics
    training_loss = tf.keras.metrics.Mean(
        "training_loss", dtype=tf.float32
    )  # We store loss here and reset after every global steps
    learning_rate_holder = tf.keras.metrics.Mean(
        "learning_rate_holder", dtype=tf.float32
    )  # We store learning rate here and reset after every global steps
    validation_loss = tf.keras.metrics.Mean(
        "validation_loss", dtype=tf.float32
    )
    
    # dataset to iterator
    dataset_iterator = iter(dataset.repeat(epochs+1))
    training_loss_holder   = []
    learning_rate_holder_history = []
    validation_loss_holder = []
    validation_score       = []
    validation_steps = []
    
    # Do before model got trained
    if validation_dataset:
        val_result = do_validation(validation_dataset)
        validation_loss_holder.append(val_result['val_loss'])
        validation_score.append(val_result['val_score'])
        validation_steps.append(0)
            
    history = {}
    for epoch in range(epochs):
        epoch_loss = []
        for step in range(steps_per_epoch // steps_per_call):

            steps_covered = (step+1) * steps_per_call
            start_time = time.time()
            train(dataset_iterator)
            end_time   = time.time()
            epoch_loss.append(training_loss.result())
            training_loss.reset_states()
            learning_rate_holder_history.append(learning_rate_holder.result())
            learning_rate_holder.reset_states()
            print("Epoch {} --- Step {}/{} --- LR --- {} Loss {} --- Time {} seconds ".format(epoch+1, 
                                                               steps_covered, 
                                                               steps_per_epoch,
                                                               learning_rate_holder_history[-1], 
                                                               epoch_loss[-1], 
                                                               end_time-start_time), end="\r")
            # Do after provided steps
            if validation_interval_steps:
                if steps_covered % validation_interval_steps == 0:
                    start_time = time.time()
                    val_result = do_validation(validation_dataset)
                    end_time = time.time()
                    validation_loss_holder.append(val_result['val_loss'])
                    validation_score.append(val_result['val_score'])
                    validation_steps.append(steps_covered)
                    print("Epoch {} --- validation Step {} --- Loss {} --- eval score {} Time {} seconds ".format(epoch, 
                                                                    steps_covered, 
                                                                    steps_per_epoch, 
                                                                    validation_loss_holder[-1], 
                                                                    validation_score[-1], 
                                                                    end_time-start_time), end="\r")
                    manager.save()
            training_loss_holder.extend(epoch_loss)
            print("Epoch {} --- mean Loss".format(epoch+1, tf.reduce_mean(epoch_loss)))
        # Do after every epoch
        if validation_dataset:
            val_result = do_validation(validation_dataset)
            validation_loss_holder.append(val_result['val_loss'])
            validation_score.append(val_result['val_score'])
            validation_steps.append(steps_covered)
        manager.save()

    history["training_loss"] = training_loss_holder
    history["val_loss"] = validation_loss_holder
    history["val_score"] = validation_score
    history["val_steps"] = validation_steps
    history["learning_rate"] = learning_rate_holder_history
    
    return history