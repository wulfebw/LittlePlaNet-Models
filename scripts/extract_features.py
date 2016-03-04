import caffe
import numpy as np
import os
import sys

model_prototxt = '../features/deploy.prototxt'
model_trained = '../models/best_citynet/_iter_100000.caffemodel'
mean_path = '../data/mean_image.binaryproto'
layer_name = 'pool5/7x7_s1'
images_filepath = '../data/train.txt'
features_filepath = '../features/features.txt'
labels_filepath = '../features/label.txt'
output_image_filepaths = '../features/image_filepaths.txt'

def crop_center(img):
    return img[16:-16,16:-16,:]

def main():
    caffe.set_mode_gpu()
    
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_path).read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    mean = arr[0][:,16:-16,16:-16]

    print mean.shape
    raw_input()
    net = caffe.Classifier(model_prototxt, model_trained,
                          mean=mean,
                          channel_swap=(2,1,0),
                          raw_scale=255,
                          image_dims=(256, 256))


    labels = []
    filepaths =[]
    count = 0
    with open(images_filepath, 'r') as reader:
        with open(features_filepath, 'w') as fw:
            for line in reader:
                count += 1
                if count > 1000:
                    break
                image_path, label = line.strip().split(' ')
                labels.append(label)
                filepaths.append(image_path)
                print 'extracting from: {}'.format(image_path)
                input_image = crop_center(caffe.io.load_image(image_path))
                prediction = net.predict([input_image], oversample=False)
                np.savetxt(fw, net.blobs[layer_name].data[0].reshape(1,-1), fmt='%.5g')

    with open(labels_filepath, 'w') as writer:
        writer.writelines(labels)

    with open(output_image_filepaths, 'w') as writer:
        writer.writelines(filepaths)

if __name__ == "__main__":
    main()
