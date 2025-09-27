import json

def simplify_encodings(input_path, output_path):
    """
    Reads an AIMET encodings file, extracts bitwidth, min, and max values,
    and saves the simplified data to a new JSON file.

    Args:
        input_path (str): Path to the input encodings file.
        output_path (str): Path to save the simplified output JSON file.
    """
    with open(input_path, 'r') as f:
        encodings = json.load(f)

    simplified_encodings = {}

    for top_level_key, layer_encodings in encodings.items():
        if isinstance(layer_encodings, dict):
            simplified_encodings[top_level_key] = {}
            for layer_name, encoding_list in layer_encodings.items():
                simplified_list = []
                if isinstance(encoding_list, list):
                    for encoding_dict in encoding_list:
                        simplified_dict = {
                            'bitwidth': encoding_dict.get('bitwidth'),
                            'min': encoding_dict.get('min'),
                            'max': encoding_dict.get('max')
                        }
                        simplified_list.append(simplified_dict)
                simplified_encodings[top_level_key][layer_name] = simplified_list

    with open(output_path, 'w') as f:
        json.dump(simplified_encodings, f, indent=4)

    print(f"Successfully processed {input_path}")
    print(f"Simplified encodings saved to {output_path}")

if __name__ == '__main__':
    # --- 使用示例 ---
    # 您需要将下面的输入文件路径替换为您自己的文件路径
    input_file = '/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/aimet_log/resnet18_adaround/adaround_2025-09-27-18-59-03/adaround_resnet.encodings'
    
    # 定义输出文件路径
    output_file = '/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/aimet_log/resnet18_adaround/adaround_2025-09-27-18-59-03/adaround_resnet_simplified.json'

    # 调用函数进行处理
    simplify_encodings(input_file, output_file)
