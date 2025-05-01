from openai import OpenAI
import base64
import os
import sys
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm  # For progress indication
import glob

def process_image(client, image_path, output_path, question, provider_name, provider_model, timeout, overwrite=False):
    """Process a single image with the AI model and save results to a JSON file."""
    # Check if output file exists and handle accordingly
    output_path_obj = Path(output_path)
    if output_path_obj.exists() and not overwrite:
        # Find a new filename with incremental suffix
        counter = 2
        while True:
            new_output_path = output_path_obj.with_stem(f"{output_path_obj.stem}-{counter}")
            if not new_output_path.exists():
                output_path = str(new_output_path)
                break
            counter += 1
    
    # Read and encode image
    print(f"Reading image from {image_path}...")
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        return False
    
    # Make API request with progress indication
    print(f"Sending request to {provider_name} ({provider_model})...")
    print("This may take a while. Please wait...")
    
    # Create a progress spinner
    start_time = time.time()
    try:
        # Send request with timeout
        response = client.chat.completions.create(
            model=provider_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ],
            max_tokens=4000,
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        print(f"Response received in {elapsed_time:.2f} seconds")
        
        response_text = response.choices[0].message.content
        print("\nResponse from model:")
        print(response_text)
        
        # Try to parse the response as JSON
        try:
            # Check if the response is already in valid JSON format
            json_data = json.loads(response_text)
            print("Response successfully parsed as JSON")
        except json.JSONDecodeError:
            # If not, try to extract JSON from the text
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(1))
                    print("JSON extracted and parsed from response text")
                except json.JSONDecodeError:
                    # If we still can't parse it, save the raw response
                    json_data = {"raw_response": response_text, 
                                "error": "Could not parse response as JSON"}
                    print("WARNING: Could not parse response as JSON")
            else:
                json_data = {"raw_response": response_text, 
                            "error": "Could not extract JSON from response"}
                print("WARNING: Could not extract JSON from response")
        
        # Save JSON output to file
        with open(output_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
        
        print(f"Analysis results saved to {output_path}")
        return True
        
    except TimeoutError:
        print(f"Error: Request timed out after {timeout} seconds")
        # Save error information to output file
        with open(output_path, 'w') as json_file:
            json.dump({"error": f"Request timed out after {timeout} seconds"}, json_file, indent=2)
        print(f"Error information saved to {output_path}")
        return False
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        # Save cancellation information to output file
        with open(output_path, 'w') as json_file:
            json.dump({"error": "Operation cancelled by user"}, json_file, indent=2)
        print(f"Cancellation information saved to {output_path}")
        raise KeyboardInterrupt
        
    except Exception as e:
        print(f"Error: {e}")
        # Save error information to output file
        with open(output_path, 'w') as json_file:
            json.dump({"error": str(e)}, json_file, indent=2)
        print(f"Error information saved to {output_path}")
        return False

def process_folder(folder_path, client, question, provider_name, provider_model, timeout, overwrite=False):
    """Process all JPG files in the specified folder."""
    # Find all JPG files in the folder
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
    image_paths.extend(glob.glob(os.path.join(folder_path, '*.jpeg')))
    
    if not image_paths:
        print(f"No JPG files found in {folder_path}")
        return
    
    print(f"Found {len(image_paths)} JPG files to process")
    
    # Process each image with progress bar
    for img_path in tqdm(image_paths, desc="Processing images"):
        output_path = Path(img_path).with_suffix('.json')
        try:
            process_image(
                client=client, 
                image_path=img_path, 
                output_path=str(output_path), 
                question=question, 
                provider_name=provider_name, 
                provider_model=provider_model, 
                timeout=timeout,
                overwrite=overwrite
            )
        except KeyboardInterrupt:
            print(f"Stopped processing at {img_path}")
            raise
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze Drosophila courtship behavior in images')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to a single image file to analyze')
    group.add_argument('--folder', type=str, help='Path to a folder containing JPG images to analyze')
    
    parser.add_argument('--provider', type=str, default='ollama', choices=['openrouter', 'ollama'],
                        help='AI provider to use (default: ollama)')
    parser.add_argument('--query', type=str, default='query.txt', 
                        help='Path to the query text file (default: query.txt)')
    parser.add_argument('--output', type=str, 
                        help='Path to save JSON output (default: same location as image with .json extension)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Request timeout in seconds (default: 60)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing JSON files instead of creating new ones with suffixes')
    
    args = parser.parse_args()
    
    providers = {
        'openrouter': {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': '<REDACTED>',
            'model': 'meta-llama/llama-3.2-90b-vision-instruct'
        },
        'ollama': {
            'base_url': 'http://ollama.vpn.gilest.ro/v1/',
            'api_key': 'ollama',
            'model': 'llama3.2'
        },
    }
    
    provider_name = args.provider
    
    # Verify query file exists
    if not os.path.exists(args.query):
        print(f"Error: Query file not found: {args.query}")
        sys.exit(1)
    
    # Check if provider is valid
    if provider_name not in providers:
        print(f"Error: Unknown provider: {provider_name}")
        print(f"Available providers: {', '.join(providers.keys())}")
        sys.exit(1)
    
    # Read query from file
    print(f"Reading query from {args.query}...")
    try:
        with open(args.query, "r") as query_file:
            question = query_file.read()
    except Exception as e:
        print(f"Error reading query file: {e}")
        sys.exit(1)
    
    try:
        # Initialize the OpenAI client with selected provider
        client = OpenAI(
            base_url=providers[provider_name]['base_url'],
            api_key=providers[provider_name]['api_key'],
            timeout=args.timeout,
        )
        
        if args.image:
            # Process single image
            image_path = args.image
            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                sys.exit(1)
                
            # Determine output file path
            if args.output:
                output_path = args.output
            else:
                # Create output path based on input image path
                output_path = str(Path(image_path).with_suffix('.json'))
                
            process_image(
                client=client, 
                image_path=image_path, 
                output_path=output_path, 
                question=question, 
                provider_name=provider_name, 
                provider_model=providers[provider_name]['model'], 
                timeout=args.timeout,
                overwrite=args.overwrite
            )
        
        elif args.folder:
            # Process all images in folder
            folder_path = args.folder
            if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                print(f"Error: Folder not found: {folder_path}")
                sys.exit(1)
                
            process_folder(
                folder_path=folder_path,
                client=client,
                question=question,
                provider_name=provider_name,
                provider_model=providers[provider_name]['model'],
                timeout=args.timeout,
                overwrite=args.overwrite
            )
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
