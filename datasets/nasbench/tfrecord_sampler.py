import argparse
import tensorflow as tf
import tqdm


def _sample_and_write_tfrecord(n, input_path, output_path):
    with tf.python_io.TFRecordWriter(path=output_path) as f:
        it = tf.python_io.tf_record_iterator(input_path)
        for i, row in tqdm.tqdm(enumerate(it), total=n):
            if i >= n:
                break
            f.write(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Samples part of input TFrecord')
    parser.add_argument('-n', '--number_of_rows', type=int,
                        required=True, help='Rows of TFrecord to be saved')
    parser.add_argument('-i', '--input_tfrecord_path', required=True,
                        help='TFRecord to be sampled')
    parser.add_argument('-o', '--output_tfrecord_path',
                        required=True, help='Output TFRecord path')
    args = parser.parse_args()

    _sample_and_write_tfrecord(
        args.number_of_rows, args.input_tfrecord_path, args.output_tfrecord_path)
