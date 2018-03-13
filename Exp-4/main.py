# -*- coding: utf-8 -*-

import os
import itertools
import re
import datetime
import editdistance
import numpy as np
import pylab
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
import keras.callbacks
import wordgenerators_sequential as wg
from sys import getdefaultencoding
import sys
import model as Model
import HelperFunctions as hp
from TextImageGenerator import TextImageGenerator

d = getdefaultencoding()
if d != "utf-8":
    reload(sys)
    sys.setdefaultencoding("utf-8")
OUTPUT_DIR = 'image_ocr'


np.random.seed(55)

def decode_batch(test_func, word_batch):

    Total = wg.getTotalData()

    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        print(wg.decodeNewDataset(out_best))
        ret.append(wg.decodeNewDataset(out_best))
        # print(wg.decodeNewDataset(out_best))
    return ret


class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()



def train(run_name, start_epoch, stop_epoch, img_w,type_t):

    img_h = 64
    pool_size = 2
    minibatch_size = 32

    fdir = os.path.dirname(get_file('wordlists.tgz',
                                    origin='http://www.mythic-ai.com/datasets/wordlists.tgz', untar=True))
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))
    img_gen = TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                 bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 2),
                                 val_split=words_per_epoch - val_words,
                                 type_t=type_t
                                 )

    model,input_data, y_pred = Model.model(img_h,img_w)

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model.compile(loss=[lambda y_true, y_pred: y_pred,'mse'], optimizer=sgd,metrics=['accuracy'])

    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())
    # model.load_weights('weightswithresize.h5')

    history = model.fit_generator(generator=img_gen.next_train(),
                        steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
                        epochs=stop_epoch,
                        validation_data=img_gen.next_val(),
                        validation_steps=val_words // minibatch_size,
                        callbacks=[viz_cb, img_gen],
                        initial_epoch=start_epoch)



if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    type_t="other"
    train(run_name, 0,200, 564,type_t)
