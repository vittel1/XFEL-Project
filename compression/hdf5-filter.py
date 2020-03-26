import sys
import numpy as np
import h5py
import hdf5plugin
import datetime
import os
import csv
import math
import random
import shutil


def getImageData(path, image_data_path):
    with h5py.File(path, 'r') as f:
        image_data = f[image_data_path][()]
        return image_data


def getMaskData(path, mask_path):
    with h5py.File(path, 'r') as f:
        mask_data = f[mask_path][()]
        return mask_data


def getGainData(path, gain_path):
    with h5py.File(path, 'r') as f:
        gain_data = f[gain_path][()]
        return gain_data


#path for raw
#/gpfs/exfel/exp/XMPL/201750/p700000/raw/r0001/
#file: RAW-R0017-AGIPD00-S00003.h5

#path for proc
#/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0001
#file: CORR-R0017-AGIPD00-S00003.h5

file_dir_path = 'files/'
no_compressed_file_path = 'compressed_files/no/'
lz4_compressed_file_path = 'compressed_files/lz4/'
zstd_compressed_file_path = 'compressed_files/zstd/'
blosc_compressed_file_path = 'compressed_files/blosc/'
scratch_path = '/gpfs/exfel/data/scratch/mpape/'

# file information for raw file
#raw_file_name = 'RAW-R0017-AGIPD00-S00000.h5'
raw_file_name = 'RAW-R0017-AGIPD00-S00003.h5'
proc_file_name = 'CORR-R0017-AGIPD00-S00003.h5'
image_data_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/data'
image_gain_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/gain'
image_mask_path = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/mask'

image_data_raw = getImageData(file_dir_path + raw_file_name, image_data_path)
image_data_proc = getImageData(file_dir_path + proc_file_name, image_data_path)

image_gain_proc = getGainData(file_dir_path + proc_file_name, image_gain_path)
image_mask_proc = getMaskData(file_dir_path + proc_file_name, image_mask_path)


def removeFile(current_path, file_name):
    '''
    Removes a file from its current path
    '''
    os.remove(current_path + file_name)


def decompressDictToCSV(dict):
    filename = 'decompression.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as f:
        w = csv.DictWriter(f, dict.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(dict)


def decompress(dset, data_shape, chunk_shape, bit_type):
    '''
    Decompressed a entire data set.
    '''
    time_list = []
    #raw data
    if len(chunk_shape) == 4:
        for i in range(1,11):
            start = datetime.datetime.now()
            data_set = dset[()]
            end = datetime.datetime.now()
            #In ms
            total_time_s = (end - start).total_seconds()
            time_list.append(total_time_s)
    #proc data
    else:
        
        for i in range(1,11):
            start = datetime.datetime.now()
            data_set = dset[()]
            end = datetime.datetime.now()
            #In ms
            total_time_s = (end - start).total_seconds()
            time_list.append(total_time_s)
    
    #size in bytes of shape
    chunk_size_byte = np.prod(np.array(list(chunk_shape))) * bit_type / 8
    #mean for decompression
    mean_time_s = sum(time_list) / len(time_list)
    
    resultList = [mean_time_s, data_set.shape, chunk_size_byte]
    return resultList


def createDict(algo_name, chunk_size_byte, compressed_file_size_bytes, data, dset, file_to_compress, image_data_size,
               mb_per_s, ratio, total_time_ms, decomp_time, decomp_shape, decomp_chunk_size):
    dict = {
        "compression_algo": algo_name,
        "file_name": file_to_compress,
        "image_data_shape": data.shape,
        "image_data_dtype": data.dtype,
        "image_data_size": image_data_size,
        "compressed_size": compressed_file_size_bytes,
        "ratio": "{0:.5f}".format(ratio),
        "chunks_shape_for_compression": dset.chunks,
        "chunk_size_byte": chunk_size_byte,
        "time_to_compress": "{0:.2f}".format(total_time_ms),
        "comp_mb/s": "{0:.2f}".format(mb_per_s),
        "decompression_time": "{0:.2f}".format(decomp_time),
        "decomp_mb/s": "{0:.2f}".format((image_data_size / (1000 * 1000)) / decomp_time),
        "decompression_shape": decomp_shape,
        "decompression_chunk_size": decomp_chunk_size,
    }
    return dict


def compress_lz4(file_to_compress, file_name_output, dataset_name, data, data_type, bit_type, comp_algo, chunk):
    '''
    Compression function for lz4
    '''
    with h5py.File(lz4_compressed_file_path + file_name_output, 'w') as f:
        time_list = []
        for i in range(1,11):
            start = datetime.datetime.now()
            dset = f.create_dataset(dataset_name + str(i),
                                    data=data,
                                    dtype=data_type,
                                    chunks=chunk,
                                    **hdf5plugin.LZ4())
            end = datetime.datetime.now()
            #In ms
            total_time_s = (end - start).total_seconds()
            time_list.append(total_time_s)
        
        mean_time_s = sum(time_list) / len(time_list)
        #In image_data_size
        image_data_size = sys.getsizeof(data)
        #Out file size in bytes
        compressed_file_size_bytes = os.path.getsize(lz4_compressed_file_path + file_name_output) / 10
        #ratio
        ratio = compressed_file_size_bytes / image_data_size
        #chunk size in bytes
        chunk_size_byte = np.prod(np.array(list(dset.chunks))) * bit_type / 8
        #mb/s
        mb_per_s = (image_data_size / (1000 * 1000)) / mean_time_s

        result_decomp = decompress(dset, data.shape, dset.chunks, bit_type)

        #create dict
        result_dict = createDict('lz4', chunk_size_byte, compressed_file_size_bytes, data, dset, file_to_compress,
                                 image_data_size, mb_per_s, ratio, mean_time_s, result_decomp[0], result_decomp[1], result_decomp[2])

        return result_dict


def compress_zstd(file_to_compress, file_name_output, dataset_name, data, data_type, bit_type, comp_algo, chunk):
    '''
    Compression function for zstd
    '''
    with h5py.File(zstd_compressed_file_path + file_name_output, 'w') as f:
        time_list = []
        for i in range(1,11):
            start = datetime.datetime.now()
            dset = f.create_dataset(dataset_name + str(i),
                                    data=data,
                                    dtype=data_type,
                                    compression=comp_algo,
                                    chunks=chunk)
            end = datetime.datetime.now()
            #In ms
            total_time_s = (end - start).total_seconds()
            time_list.append(total_time_s)
        
        mean_time_s = sum(time_list) / len(time_list)
        #In image_data_size
        image_data_size = sys.getsizeof(data)
        #Out file size in bytes
        compressed_file_size_bytes = os.path.getsize(zstd_compressed_file_path + file_name_output) / 10
        #ratio
        ratio = compressed_file_size_bytes / image_data_size
        #chunk size in bytes
        chunk_size_byte = np.prod(np.array(list(dset.chunks))) * bit_type / 8

        #mb/s
        mb_per_s = (image_data_size / (1000 * 1000)) / mean_time_s

        result_decomp = decompress(dset, data.shape, dset.chunks, bit_type)

        #create dict
        result_dict = createDict('zstd', chunk_size_byte, compressed_file_size_bytes, data, dset, file_to_compress,
                                 image_data_size, mb_per_s, ratio, mean_time_s, result_decomp[0], result_decomp[1], result_decomp[2])

        return result_dict


def compress_blosc(file_to_compress, file_name_output, dataset_name, data, data_type, bit_type, comp_algo, chunk):
    '''
    Compression function for blosc
    '''
    with h5py.File(blosc_compressed_file_path + file_name_output, 'w') as f:
        time_list = []
        for i in range(1,11):
            start = datetime.datetime.now()
            dset = f.create_dataset(dataset_name + str(i),
                                    data=data,
                                    dtype=data_type,
                                    chunks=chunk,
                                    **hdf5plugin.Blosc(cname='blosclz', shuffle=hdf5plugin.Blosc.NOSHUFFLE))
            end = datetime.datetime.now()
            #In ms
            total_time_s = (end - start).total_seconds()
            time_list.append(total_time_s)
        
        mean_time_s = sum(time_list) / len(time_list)
        #In image_data_size
        image_data_size = sys.getsizeof(data)
        #Out file size in bytes
        compressed_file_size_bytes = os.path.getsize(blosc_compressed_file_path + file_name_output) / 10
        #ratio
        ratio = compressed_file_size_bytes / image_data_size
        #chunk size in bytes
        chunk_size_byte = np.prod(np.array(list(dset.chunks))) * bit_type / 8
        #mb/s
        mb_per_s = (image_data_size / (1000 * 1000)) / mean_time_s

        result_decomp = decompress(dset, data.shape, dset.chunks, bit_type)

        #create dict
        result_dict = createDict('blosc', chunk_size_byte, compressed_file_size_bytes, data, dset, file_to_compress,
                                 image_data_size, mb_per_s, ratio, mean_time_s, result_decomp[0], result_decomp[1], result_decomp[2])

        return result_dict



def compressDictToCSV(dict):
    '''
    Write the result dict to a csv file.
    The first value for each dict is the default shape choosen by hdf5.
    Header will only be written ones.
    '''
    filename = 'compression.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as f:
        w = csv.DictWriter(f, dict.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(dict)


def lz4(file_name, image_data, type_of_input_file, dtype, bit_type, chunkList):
    '''
    Calls the lz4 compression function for each different chunksize
    '''
    LZ4_COMPRESS_VALUE = 32004

    result_compressed_list = []
    file_prefix = 0
    for chunk in chunkList:
        lz4_compressed_file_name = 'lz4_{}_{}_compressed.h5'.format(file_prefix, type_of_input_file)
        file_prefix = file_prefix + 1
        compressed_lz4 = compress_lz4(file_name, lz4_compressed_file_name, 'data', image_data, dtype, bit_type, LZ4_COMPRESS_VALUE, chunk)
        result_compressed_list.append(compressed_lz4)
        removeFile(lz4_compressed_file_path, lz4_compressed_file_name)

    #result to csv
    for result_dict in result_compressed_list:
        compressDictToCSV(result_dict)


def zstd(file_name, image_data, type_of_input_file, dtype, bit_type, chunkList):
    '''
    Calls the zstd function for each different chunksize
    '''
    ZSTD_COMPRESS_VALUE = 32015

    result_compressed_list = []
    file_prefix = 0
    for chunk in chunkList:
        zstd_compressed_file_name = 'zstd_{}_{}_compressed.h5'.format(file_prefix, type_of_input_file)
        file_prefix = file_prefix + 1
        compressed_zstd = compress_zstd(file_name, zstd_compressed_file_name, 'data', image_data, dtype, bit_type, ZSTD_COMPRESS_VALUE, chunk)
        result_compressed_list.append(compressed_zstd)
        removeFile(zstd_compressed_file_path, zstd_compressed_file_name)

    #result to csv
    for result_dict in result_compressed_list:
        compressDictToCSV(result_dict)


def blosc(file_name, image_data, type_of_input_file, dtype, bit_type, chunkList):
    '''
    Calls the blosc function for each different chunksize
    '''
    BLOSC_COMPRESS_VALUE = 32001

    result_compressed_list = []
    file_prefix = 0
    for chunk in chunkList:
        blosc_compressed_file_name = 'blosc_{}_{}_compressed.h5'.format(file_prefix, type_of_input_file)
        file_prefix = file_prefix + 1
        compressed_blosc = compress_blosc(file_name, blosc_compressed_file_name, 'data', image_data, dtype, bit_type, BLOSC_COMPRESS_VALUE, chunk)
        result_compressed_list.append(compressed_blosc)
        removeFile(blosc_compressed_file_path, blosc_compressed_file_name)

    #result to csv
    for result_dict in result_compressed_list:
        compressDictToCSV(result_dict)


#one puls: (1,x,512,128)
def calculateRawChunk():
    shapeList = []
    for i in range(1, 11):
        shapeList.append((i, 1, 512, 128))
    # for default value
    shapeList.append(True)
    return shapeList


def calculateProcChunk():
    shapeList = []
    for i in range(1, 11):
        shapeList.append((i, 512, 128))
    #for default value
    shapeList.append(True)
    return shapeList


chunk_list_raw = calculateRawChunk()
chunk_list_proc = calculateProcChunk()

bit_type_raw = 16
bit_type_proc = 32
bit_type_gain_mask = 8

#raw
lz4(raw_file_name, image_data_raw, 'raw', np.uint16, bit_type_raw, chunk_list_raw)
zstd(raw_file_name, image_data_raw, 'raw', np.uint16, bit_type_raw, chunk_list_raw)
blosc(raw_file_name, image_data_raw, 'raw', np.uint16, bit_type_raw, chunk_list_raw)

#proc
lz4(proc_file_name, image_data_proc, 'proc', np.float32, bit_type_proc, chunk_list_proc)
zstd(proc_file_name, image_data_proc, 'proc', np.float32, bit_type_proc, chunk_list_proc)
blosc(proc_file_name, image_data_proc, 'proc', np.float32, bit_type_proc, chunk_list_proc)


#compression for gain in proc
#blosc(proc_file_name, image_gain_proc, 'proc_gain', np.int8, bit_type_gain_mask, chunk_list_proc)
#compression for mask in proc
#blosc(proc_file_name, image_mask_proc, 'proc_mask', np.int8, bit_type_gain_mask, chunk_list_proc)