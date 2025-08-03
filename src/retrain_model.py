# src/retrain_model.py

from tensorflow.keras.callbacks import EarlyStopping
from model import build_model, save_model
from preprocessing import get_data_generators
import os

def retrain_model(train_dir="../data/train", user_dir="../data/user_uploaded", model_path="../models/fruit_model.keras", epochs=5):
    """
    Retrains the model using data from both the original and user-uploaded datasets.
    """
    # Create a temporary merged training folder
    import shutil
    merged_train_dir = "../data/_combined_train"
    if os.path.exists(merged_train_dir):
        shutil.rmtree(merged_train_dir)
    shutil.copytree(train_dir, merged_train_dir)

    # Merge user-uploaded data
    for class_dir in os.listdir(user_dir):
        src = os.path.join(user_dir, class_dir)
        dst = os.path.join(merged_train_dir, class_dir)
        os.makedirs(dst, exist_ok=True)
        for file in os.listdir(src):
            shutil.copy(os.path.join(src, file), os.path.join(dst, file))

    # Load generators
    train_gen, val_gen, test_gen = get_data_generators("../data", target_size=(224, 224), batch_size=32, train_subfolder="_combined_train")

    # Build and train model
    model = build_model(input_shape=(224, 224, 3), num_classes=len(train_gen.class_indices))
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
    )

    # Save model
    save_model(model, model_path)

    return f"Retraining complete. Model saved to {model_path}"
