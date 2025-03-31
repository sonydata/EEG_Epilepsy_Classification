import os
import re
import pandas as pd

#Parsing the directory structure to get metadata
def parse_edf_path(edf_path):

    #split path by separator  /
    parts = edf_path.split(os.sep)
    #refer to naming convention in the readme
    subject_id = parts[-4]                
    session_info = parts[-3]          
    montage = parts[-2]                
    filename = parts[-1] 
    #get token from filename              
    token = filename.split('_')[-1].split('.')[0]
    #get date and session number
    session_parts = session_info.split('_')
    session = session_parts[0]                     
    session_date = '_'.join(session_parts[1:]) if len(session_parts) > 1 else None
    
    return subject_id, session, session_date, montage, token

##Parsing one line of the .list file that contains demographic data
def parse_demo_line(line):

    tokens = line.strip().split()
    if len(tokens) < 4:
        return None  #skip if not enough tokens
    subject_id = tokens[0]
    #extract age from token like "[Age:35]"
    age_match = re.search(r'\[Age:(\d+)\]', tokens[1])
    age = int(age_match.group(1)) if age_match else None
    #extract gender from token like "[M]" or "[F]"
    gender_match = re.search(r'\[([MF])\]', tokens[2])
    gender = gender_match.group(1) if gender_match else None
    #the rest is ethnicity
    ethnicity = ' '.join(tokens[3:])
    return subject_id, age, gender, ethnicity

##Parsing all demographic info
def load_demo_file(filepath):
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            
            parsed = parse_demo_line(line)
            
            subject_id, age, gender, ethnicity = parsed
            records.append({
                        "subject_id": subject_id,
                        "age": age,
                        "gender": gender,
                        "ethnicity": ethnicity
                    })
    return pd.DataFrame(records)

##Parsing all metadata

def extract_metadata(groups):

    all_dataframes = []

    for grp in groups:
        group_dir = grp["group_dir"]
        demo_file = grp["demo_file"]
        patient_group = grp["patient_group"]
        
        print('Parsing demo file')
        print(grp)
        
        demo_df = load_demo_file(demo_file)
        
        records = []
        print('Parsing EEG metadata')
        for root, dirs, files in os.walk(group_dir): #exploring all the folders
            for file in files: 
                if file.endswith('.edf'):
                    edf_path = os.path.join(root, file)
                    subject_id, session, session_date, montage, token = parse_edf_path(edf_path)
                    records.append({
                        "edf_path": edf_path,
                        "subject_id": subject_id,
                        "session": session,
                        "session_date": session_date,
                        "montage": montage,
                        "token": token,
                        "patient_group": patient_group
                    })
        
        edf_df = pd.DataFrame(records)
        
        merged_df = pd.merge(edf_df, demo_df, on="subject_id", how="left")
        all_dataframes.append(merged_df)
        print('Done')

    final_df = pd.concat(all_dataframes, ignore_index=True)
    print(final_df)
    final_df.to_excel('eeg_metadata.xlsx')

groups = [
    {"group_dir": "epilepsy_data/00_epilepsy", "demo_file": "epilepsy_data/00_subject_ids_epilepsy.list", "patient_group": "epilepsy"},
    {"group_dir": "epilepsy_data/01_no_epilepsy", "demo_file": "epilepsy_data/01_subject_ids_no_epilepsy.list", "patient_group": "no_epilepsy"}
]

metadata_df = extract_metadata(groups)
print(metadata_df)