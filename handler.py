import os
import numpy as np
from matplotlib import pyplot
from PIL import Image
from matplotlib.path import Path
from scipy.interpolate import interp1d
import json
import xml.etree.ElementTree as ET
import logging


annotations_path = os.path.join('database', "Annotations", "users", "lauralrh")
images_path = os.path.join('database', "Images", "users", "lauralrh")


def name_r():
    return str(np.random.random(1)[0])[2:]


def fix_right_left(data):
    w = list(data.keys())
    for key in w:
        ks = list(data[key].keys())[:]
        # print(ks)

        if len(ks) > 1:

            data[key]['right'] = data[key][str(min(list(map(int, ks))))]
            data[key]['left'] = data[key][str(max(list(map(int, ks))))]

            del data[key][str(min(list(map(int, ks))))]
            del data[key][str(max(list(map(int, ks))))]

        else:
            if int(ks[0]) > 200:
                data[key]['left'] = data[key][str(min(list(map(int, ks))))]
            else:

                data[key]['right'] = data[key][str(min(list(map(int, ks))))]

    return data


# def get_polygons(image, annotation, range_):

#     logging.info(f"Loadding: {annotation}")
#     tree = ET.parse(annotation)
#     root = tree.getroot()

#     polygons = {}

#     for obj in root.findall('object'):

#         name = obj.find('name').text
#         id_ = obj.find('id').text

#         if int(id_) < len( root.findall('object'))//2:
#             id_ = 'right'
#         else:
#             id_ = 'left'

#         polygon = []
#         for pt in obj.find('polygon').findall('pt'):
#             polygon.append([pt.find('x').text, pt.find('y').text])

#         if not name in polygons:
#             polygons[name] = {}

#         polygons[name][id_] = polygon


#     logging.info(f"Loadding: {image}")
#     img = np.array(Image.open(image))[:,:,0]
#     # pyplot.figure(figsize=(20, 10), dpi=90)
#     # pyplot.imshow(img)
#     scale_min, scale_max = range_#[os.path.split(image)[-1]]

#     logging.info(f"scalling from {scale_min}-{scale_max} to 0-255")

#     polygons_arr = {}
#     mask = np.array(list(zip(*np.mgrid[0:640, 0:480].reshape(2, -1))))
#     for name in polygons:
#         for id_ in polygons[name]:

#     #         line = pyplot.Polygon(polygons[name][id_], closed=True, color=f"C{name}", alpha=1, fill=True)
#     #         pyplot.gca().add_line(line)

#             p = Path(polygons[name][id_])
#             grid = p.contains_points(mask)

#             if not name in polygons_arr:
#                 polygons_arr[name] = {}

#             adjust_scale = interp1d([0, 255], [scale_min, scale_max])

#             polygons_arr[name][id_] = adjust_scale(np.array([img.T.item(tuple(p)) for p in mask[grid]]))

#     logging.info(f"{len(polygons_arr)} polygons")
#     logging.info(f"IDs: ", {k:len(polygons_arr[k]) for k in polygons_arr.keys()})

#     logging.info("-"*70)

#     return polygons_arr


def get_polygons(image, annotation, range_):

    logging.info(f"Loadding: {annotation}")
    tree = ET.parse(annotation)
    root = tree.getroot()

    polygons = {}

    for obj in root.findall('object'):

        name = obj.find('name').text
        id_ = obj.find('id').text

#         if int(id_) < len( root.findall('object'))//2:
#             id_ = 'right'
#         else:
#             id_ = 'left'

        id_ = name_r()

        polygon = []
        for pt in obj.find('polygon').findall('pt'):
            polygon.append([pt.find('x').text, pt.find('y').text])


#         print(pt.find('x').text)
        id_ = pt.find('x').text

#         if int(id_) < len( root.findall('object'))//2:
#             id_ = 'right'
#         else:
#             id_ = 'left'

        if not name in polygons:
            polygons[name] = {}

        polygons[name][id_] = polygon

    logging.info(f"Loadding: {image}")
    img = np.array(Image.open(image))[:, :, 0]
    # pyplot.figure(figsize=(20, 10), dpi=90)
    # pyplot.imshow(img)
    scale_min, scale_max = range_  # [os.path.split(image)[-1]]

    logging.info(f"scalling from {scale_min}-{scale_max} to 0-255")

    polygons_arr = {}
    mask = np.array(list(zip(*np.mgrid[0:640, 0:480].reshape(2, -1))))
    for name in polygons:
        for id_ in polygons[name]:

    #         line = pyplot.Polygon(polygons[name][id_], closed=True, color=f"C{name}", alpha=1, fill=True)
    #         pyplot.gca().add_line(line)

            p = Path(polygons[name][id_])
            grid = p.contains_points(mask)

            if not name in polygons_arr:
                polygons_arr[name] = {}

            adjust_scale = interp1d([0, 255], [scale_min, scale_max])

            polygons_arr[name][id_] = adjust_scale(
                np.array([img.T.item(tuple(p)) for p in mask[grid]]))

    logging.info(f"{len(polygons_arr)} polygons")
    logging.info(f"IDs: ", {k: len(polygons_arr[k])
                            for k in polygons_arr.keys()})

    logging.info("-" * 70)

#     return polygons_arr
    return fix_right_left(polygons_arr)


def get_temperatures(range_, images_path , annotations_path):

    poly = []

    file_range = False
    if isinstance(range_, str):
        ranges = image = os.path.join(images_path, f'{range_}.json')
        ranges = json.load(open(ranges, 'r'))
        file_range = True

    for file in os.listdir(images_path):
        if not file.endswith('.jpg'):
            continue

        image = os.path.join(images_path, file)
        annotation = os.path.join(
            annotations_path, file.replace('.jpg', '.xml'))

        if file_range:
            range_ = ranges[file]

        poly.append([file, get_polygons(image, annotation, range_)])

    return poly


def show_temperatures(patient, images_path, annotations_path,  fn='mean', range_='scale', sufix='', fig=None):

    if fig is None:
        fig_ = pyplot.figure(figsize=(20, 15), dpi=90)

    poly = get_temperatures(range_, images_path , annotations_path)
    pyplot.subplot(2, 3, 1)
    pyplot.title(f"{patient}{sufix}")
    pyplot.imshow(Image.open('figlabels.png'))
    pyplot.axis('off')
    Position = ['Plantar medial', 'Safeno',
                'Tibial', 'Sural', 'Plantar lateral']
    labels = ['L.4.5', 'L.3.4', 'S.1.2', 'S.1.2', 'S.1.2']
    for i in range(1, 6):
        P = str(i)

        r, l, t = [], [], []
        for t_, p in poly:

            if not P in p:
                continue

            t = int(t_.replace('.jpg', '').replace('t', ''))

            if 'right' in p[P]:
                r.append((t, getattr(np, fn)(p[P]['right'])))
            if 'left' in p[P]:
                l.append((t, getattr(np, fn)(p[P]['left'])))

        pyplot.subplot(2, 3, i + 1)
        pyplot.title(f"{Position[i-1]}: {labels[i-1]}")
        pyplot.plot(*list(zip(*sorted(r))), '-o', label='right', c='C0')
        pyplot.plot(*list(zip(*sorted(l))), '-o', label='left', c='C1')
        if fig is None:
            pyplot.legend()
        pyplot.ylabel('Temperature [°C]')
        pyplot.xlabel('Time [$s$]')
        pyplot.grid(True)
        pyplot.show()

    if fig is None:

        fig_.savefig(os.path.join('out', f"{patient}{sufix}.png"))
        pyplot.show()
