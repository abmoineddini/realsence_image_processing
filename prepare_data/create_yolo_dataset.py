
# Credits to https://github.com/computervisioneng
import os
import shutil


DATA_ALL_DIR = os.path.join('.', 'data_all')

DATA_OUT_DIR = os.path.join('.', 'data')

for set_ in ['train', 'validation', 'test']:
    for dir_ in [os.path.join(DATA_OUT_DIR, set_),
                 os.path.join(DATA_OUT_DIR, set_, 'imgs'),
                 os.path.join(DATA_OUT_DIR, set_, 'anns')]:
        if os.path.exists(dir_):
            shutil.rmtree(dir_)
        os.mkdir(dir_)

Tennis_ball_id = '/m/05ctyq'
class_names = 'tennis_ball'

train_bboxes_filename = os.path.join('.', 'data_all/boxes/oidv6-train-annotations-bbox.csv')
validation_bboxes_filename = os.path.join('.', 'data_all/boxes/validation-annotations-bbox.csv')
test_bboxes_filename = os.path.join('.', 'data_all/boxes/test-annotations-bbox.csv')


for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    set_ = ['train', 'validation', 'test'][j]
    print(j)
    print(set_)
    print(filename)
    with open(filename, 'r') as f:
        line = f.readline()
        while len(line) != 0:
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
            if class_name in [Tennis_ball_id]:
                if not os.path.exists(os.path.join(DATA_OUT_DIR, set_, 'imgs', f'{id}.jpg')):
                    shutil.copy(os.path.join(DATA_ALL_DIR,set_, class_names,f'{id}.jpg'),
                                os.path.join(DATA_OUT_DIR, set_, 'imgs', f'{id}.jpg'))
                with open(os.path.join(DATA_OUT_DIR, set_, 'anns', f'{id}.txt'), 'a') as f_ann:
                    # class_id, xc, yx, w, h
                    x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]
                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    f_ann.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                    f_ann.close()

            line = f.readline()