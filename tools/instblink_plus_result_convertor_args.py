import json
import time
from argparse import ArgumentParser
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--json',help='Path to prediction json file')
    parser.add_argument('--output',help='Path to prediction json file')
    parser.add_argument('--threshold',help='Path to prediction json file',default=0.35)
    args = parser.parse_args()
    return args



def main(args):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    eyeblink_threshold = float(args.threshold)
    results = json.load(open(args.json,'r'))
    filtered_results = []
    for query in results:
        blinks_converted = []
        eyeblink_buffer = []
        #print(query['blink_scores'])
        if 'blink_scores' in query.keys():
          for index in range(0, len(query['blink_scores'])):
              if query['blink_scores'][index] >= eyeblink_threshold and eyeblink_buffer == []:
                  eyeblink_buffer.append(index)
              if query['blink_scores'][index] < eyeblink_threshold and eyeblink_buffer != []:
                  # blinks_converted.extend([[eyeblink_buffer[0]-1, index + 1]])  # 若不扩充，则结果为[eyeblink_buffer[0], index - 1]
                  sum = 0
                  for i in range(eyeblink_buffer[0],index):       # 目前计算置信度没有考虑扩充
                      sum +=query['blink_scores'][i]
                  avg_score = sum/(index-eyeblink_buffer[0])
                  # blinks_converted.extend([[eyeblink_buffer[0] - 1, index + 2, avg_score]]) # 若不扩充，则结果为[eyeblink_buffer[0], index - 1, avg_score]
                  blinks_converted.extend([[eyeblink_buffer[0] , index - 1, avg_score]])
                  eyeblink_buffer = []
              if (index == len(query['blink_scores']) - 1) and eyeblink_buffer != []: # 如果是结束帧
                  sum = 0
                  for i in range(eyeblink_buffer[0], index+1):  # 目前计算置信度没有考虑扩充
                      sum += query['blink_scores'][i]
                  avg_score = sum / (index - eyeblink_buffer[0]+1)
                  # blinks_converted.extend([[eyeblink_buffer[0] - 1, index + 2, avg_score]]) # 若不扩充，则结果为[eyeblink_buffer[0], index - 1, avg_score]
                  blinks_converted.extend([[eyeblink_buffer[0], index, avg_score]])
                  eyeblink_buffer = []
        #print(blinks_converted)
        query.update({'blinks_converted': blinks_converted})
        filtered_results.append(query)
    
    os.makedirs('results/blink_converted_results', exist_ok=True)
    json.dump(filtered_results, open(args.output, 'w'))
    print('Done')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

if __name__ == '__main__':
    args = parse_args()
    main(args)