import torch
import argparse
import tensorflow as tf
import tqdm


def _sample_and_write_pth(n, input_path, output_path, need_loss):
    data = torch.load(input_path)

    keys = sorted(list(data['arch2infos'].keys()))
    type_list = ['less', 'full']
    epochs = [12, 200]

    for i, xkey in tqdm.tqdm(enumerate(keys)):
        if i >= n:
            break
        for idx, t in enumerate(type_list):
            all_results = data['arch2infos'][xkey][t]['all_results']
            epoch = epochs[idx]

            for key in all_results.keys():
                dataset, seed = key
                result = all_results[key]

                if isinstance(result['train_acc1es'], dict):
                    result['train_acc1es'] = max(
                        result['train_acc1es'].values())
                if isinstance(result['train_acc5es'], dict):
                    result['train_acc5es'] = max(
                        result['train_acc5es'].values())
                if isinstance(result['train_times'], dict):
                    result['train_times'] = sum(result['train_times'].values())

                for ename in result['eval_names']:
                    result['eval_acc1es'][ename] = max([result['eval_acc1es'][ename+"@"+str(
                        e)] for e in range(epoch) if ename+"@"+str(e) in result['eval_acc1es'].keys()])
                    result['eval_times'][ename] = sum([result['eval_times'][ename+"@"+str(
                        e)] for e in range(epoch) if ename+"@"+str(e) in result['eval_times'].keys()])

                    for e in range(epoch):
                        if ename+"@"+str(e) in result['eval_acc1es'].keys():
                            del result['eval_acc1es'][ename+"@"+str(e)]
                        if ename+"@"+str(e) in result['eval_times'].keys():
                            del result['eval_times'][ename+"@"+str(e)]

                if not need_loss:
                    if 'eval_losses' in result.keys():
                        del result['eval_losses']
                    if 'train_losses' in result.keys():
                        del result['train_losses']

    torch.save(data, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Samples part of input TFrecord')
    parser.add_argument('-n', '--number_of_rows', type=int,
                        required=True, help='Number of archs to be saved')
    parser.add_argument('-l', '--need_loss', type=bool, default=False,
                        required=True, help='Need loss data')
    parser.add_argument('-i', '--input_pth_path', required=True,
                        help='Pth to be sampled')
    parser.add_argument('-o', '--output_pth_path',
                        required=True, help='Output pth path',
                        default='NAS-BENCH-201-v2-wo-losses.pth')
    args = parser.parse_args()

    _sample_and_write_pth(
        args.number_of_rows, args.input_pth_path,
        args.output_pth_path, args.need_loss)
