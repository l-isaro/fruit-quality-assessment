import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, target_size=(224, 224), batch_size=32):
    """
    Creates train and test data generators from the given directory.
    
    Args:
        data_dir (str): Root directory with 'train/' and 'test/' subfolders.
        target_size (tuple): Desired image size (height, width).
        batch_size (int): Number of images per batch.

    Returns:
        train_generator, test_generator
    """
    # Paths
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Just rescaling for test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow from directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator
