from gram import*
from image_ops import*
from costs import*
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import tensorflow as tf



ratio = float(1/16)
Style_Weights = {"block1_conv1": ratio, "block1_conv2": ratio, "block2_conv1": ratio, "block2_conv2": ratio,
                 "block3_conv1": ratio, "block3_conv2": ratio, "block3_conv3": ratio, "block3_conv4": ratio,
                 "block4_conv1": ratio, "block4_conv2": ratio, "block4_conv3": ratio, "block4_conv4": ratio,
                 "block5_conv1": ratio, "block5_conv2": ratio, "block5_conv3": ratio, "block5_conv4": ratio }



def main():
    content = img_read('Van.jpg')
    style = img_read('Bogaz.jpg')
    generated = random_image(content)

    base_model = VGG19(include_top=False, weights='imagenet',input_shape=(224,224,3))
    model_cont = Model(inputs=base_model.input,outputs=base_model.get_layer('block5_conv4').output)

    custom_outputs = [base_model.get_layer(i).output for i in Style_Weights.keys()]
    model_style =  Model(inputs=base_model.input, outputs=custom_outputs)

    pred_content = model_cont.predict(vgg_input(content))
    pred_style =  model_style.predict(vgg_input(style))

    optimizer_adam = keras.optimizers.Adam(learning_rate=0.1)


    pred_generated_cont = model_cont.predict(vgg_input(generated))
    pred_generated_style = model_style.predict(vgg_input(generated))
    total_cost = cost(pred_content,pred_style,pred_generated_cont,pred_generated_style,ratio,40,10)

    ##apply grad descent to Generated

if __name__ == '__main__':
    main()
