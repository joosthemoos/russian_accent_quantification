import pandas as pd
import sys
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def general_function(sys_arg):
    #files = Path(sys.argv[1]).glob('subject*_formants.csv')
    files = Path(sys_arg).glob('*_formants.csv')
    #fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True) # for six of them
    fig, axs = plt.subplots(1, 1, figsize=(6, 6)) # for one of them
    #fig, axs = plt.subplots(1, 3, figsize=(22, 4), sharex=True, sharey=True)
    colors = {'subject_1': 'red', 'subject_2': 'orange', 'subject_4': 'green', 'subject_7': 'purple', 'CONTROL': 'blue', 'russian6': 'hotpink', 'russian7': 'firebrick', 'russian8': 'turquoise', 'russian14': 'sienna'}
    max_t_elapsed = 0
    max_t_elapsed_spa = 0
    cross_subj_info = {}
    phone_list = []
    subj_list = []
    pitch_info = {}
    for file in files:
        t_elapsed = 0
        phone_info = {}
        ifile  = open(file, encoding="utf16", errors='ignore')
        read = csv.reader(ifile)
        line_no = 0
        lv_p_min = None
        lv_p_avg = None
        subject_name = None
        subj__tot_lapsed = 0
        for line in read:
            line_no += 1
            print(line_no)
            if line_no == 1:
                continue
            print(line)
            t_elapsed = float(line[1])
            if line_no == 2:
                subject_name = line[0]
                if subject_name == "subject_3":
                    subject_name = "CONTROL"
                cross_subj_info[subject_name] = {}
                pitch_info[subject_name] = {}
                pitch_info[subject_name]['pitches'] = []
                pitch_info[subject_name]['durs'] = []
                subj_list.append(subject_name)
            if not line[3]:
                continue
            phone = line[3]
            p_min = line[9]
            p_avg = line[8]
            f1 = line[5]
            f2 = line[6]
            f3 = line[7]

            # fixing undefined minimum pitch, setting to previous valid F0
            if p_min == "--undefined--":
                if not lv_p_min:
                    continue
                else:
                    p_min = lv_p_min
                if not lv_p_avg:
                    continue
                else:
                    p_avg = lv_p_avg
            else:
                lv_p_min = p_min
                lv_p_avg = p_avg
            pitch_info[subject_name]['pitches'].append(float(p_min))
            pitch_info[subject_name]['durs'].append(t_elapsed)

            if line[3] not in phone_info:
                phone_info[line[3]] = {}
                phone_info[line[3]]['p_min'] = [float(p_min)]
                phone_info[line[3]]['p_avg'] = [float(p_avg)]
                phone_info[line[3]]['f1'] = [float(f1)]
                phone_info[line[3]]['f2'] = [float(f2)]
                phone_info[line[3]]['f3'] = [float(f3)]
            else:
                phone_info[line[3]]['p_min'].append(float(p_min))
                phone_info[line[3]]['f1'].append(float(f1))
                phone_info[line[3]]['f2'].append(float(f2))
                phone_info[line[3]]['f3'].append(float(f3))

            if phone not in cross_subj_info[subject_name]:
                print(phone, "is not in cross_subj_info[", subject_name, "]")
                cross_subj_info[subject_name][phone] = {}
                cross_subj_info[subject_name][phone]['p_min'] = [float(p_min)]
                cross_subj_info[subject_name][phone]['p_avg'] = [float(p_avg)]
                cross_subj_info[subject_name][phone]['f1'] = [float(f1)]
                cross_subj_info[subject_name][phone]['f2'] = [float(f2)]
                cross_subj_info[subject_name][phone]['f3'] = [float(f3)]
            else:
                cross_subj_info[subject_name][phone]['p_min'].append(float(p_min))
                cross_subj_info[subject_name][phone]['f1'].append(float(f1))
                cross_subj_info[subject_name][phone]['f2'].append(float(f2))
                cross_subj_info[subject_name][phone]['f3'].append(float(f3))

            if line[3] not in phone_list:
                phone_list.append(line[3])
            t_elapsed += float(line[4])
            subj_total_elapsed = t_elapsed
            if "russian" in subject_name:
                if t_elapsed > max_t_elapsed_spa:
                    max_t_elapsed_spa = t_elapsed
            else:
                if t_elapsed > max_t_elapsed:
                    max_t_elapsed = t_elapsed                
        pitch_info[subject_name]['total_dur'] = subj_total_elapsed
        a_f1 = np.array(phone_info['æ']['f1'])
        a_f2 = np.array(phone_info['æ']['f2'])
        normed_a_f1 = np.subtract(np.array(phone_info['æ']['f1']), np.array(phone_info['æ']['p_min']))
        normed_a_f2 = np.subtract(np.array(phone_info['æ']['f2']), np.array(phone_info['æ']['p_min']))
        normed_avg_a_f1 = np.subtract(np.array(phone_info['æ']['f1']), np.array(phone_info['æ']['p_avg']))
        normed_avg_a_f2 = np.subtract(np.array(phone_info['æ']['f2']), np.array(phone_info['æ']['p_avg']))
        normed_bigi_f1 = np.subtract(np.array(phone_info['ɪ']['f1']), np.array(phone_info['ɪ']['p_min']))
        normed_bigi_f2 = np.subtract(np.array(phone_info['ɪ']['f2']), np.array(phone_info['ɪ']['p_min']))
        normed_uh_f1 = np.subtract(np.array(phone_info['ʌ']['f1']), np.array(phone_info['ʌ']['p_min']))
        normed_uh_f2 = np.subtract(np.array(phone_info['ʌ']['f2']), np.array(phone_info['ʌ']['p_min']))
        normed_eps_f1 = np.subtract(np.array(phone_info['ɛ']['f1']), np.array(phone_info['ɛ']['p_min']))
        normed_eps_f2 = np.subtract(np.array(phone_info['ɛ']['f2']), np.array(phone_info['ɛ']['p_min']))
        normed_i_f1 = np.subtract(np.array(phone_info['i']['f1']), np.array(phone_info['i']['p_min']))
        normed_i_f2 = np.subtract(np.array(phone_info['i']['f2']), np.array(phone_info['i']['p_min']))
        normed_u_f1 = np.subtract(np.array(phone_info['u']['f1']), np.array(phone_info['u']['p_min']))
        normed_u_f2 = np.subtract(np.array(phone_info['u']['f2']), np.array(phone_info['u']['p_min']))
        if 'd' not in phone_info:
            continue
        #normed_d_f1 = np.subtract(np.array(phone_info['d']['f1']), np.array(phone_info['d']['p_min']))
        #normed_d_f2 = np.subtract(np.array(phone_info['d']['f2']), np.array(phone_info['d']['p_min']))
        normed_a_f1 = np.subtract(np.array(phone_info['æ']['f1']), np.array(phone_info['æ']['p_min']))
        normed_a_f2 = np.subtract(np.array(phone_info['æ']['f2']), np.array(phone_info['æ']['p_min']))

    #    bigi_f1 = np.array(phone_info['ɪ']['f1'])
    #    bigi_f2 = np.array(phone_info['ɪ']['f2'])
        if subject_name == "subject_3":
            subject_name = "CONTROL"
        #print(subject_name, normed_bigi_f1)
        axs.scatter(normed_a_f1, normed_a_f2, label=subject_name, color=colors[subject_name])
        axs.title.set_text('[æ]')
        axs.set_xlabel('F1 value (Hz)')
        axs.set_ylabel('F2 value (Hz)')
    plt.legend()
    plt.show()
    
    plot_f0s(pitch_info, max_t_elapsed, max_t_elapsed_spa)
    clean_formant_dict = clean_formants_please(cross_subj_info, subj_list, phone_list)
    print_euclids(clean_formant_dict, subj_list, phone_list)

    perceived_list = [2, 2.076923077, 2.4347826094, 3.269230769, 4.30769231]
    plot_ash_e_perceived(perceived_list, clean_formant_dict, colors)
    consonant_perceived_strength(perceived_list)

    # plotting the variances in big i
    fig, ax = plt.subplots()
    for subject_name in subj_list:
        if 'ɪ' not in clean_formant_dict[subject_name].keys():
            continue
        #import pdb; pdb.set_trace()
        #if not ("subject" in subject_name or "CONTROL" in subject_name):
        #    print("didnt work")
        #    continue
        plt.scatter(clean_formant_dict[subject_name]['ɪ']['f1_var'], clean_formant_dict[subject_name]['ɪ']['f3_var'], label=subject_name, color=colors[subject_name])
    ax.set_title('[ɪ] - Variance')
    ax.set_xlabel('F2 variance (Hz)')
    ax.set_ylabel('F3 variance (Hz)')
    #ax.set_xlim(0,6000)
    #ax.set_ylim(0,6000)
    plt.legend()
    #fig.suptitle("Regular formant vs. formant normalized with P_min")
    print("random phones about to show!")
    plt.show()

    # plotting the variances in little i
    fig, ax = plt.subplots()
    for subject_name in subj_list:
        if 'ɪ' not in clean_formant_dict[subject_name].keys():
            continue
        #import pdb; pdb.set_trace()
        #if not ("subject" in subject_name or "CONTROL" in subject_name):
        #    print("didnt work")
        #    continue
        plt.scatter(clean_formant_dict[subject_name]['i']['f2_var'], clean_formant_dict[subject_name]['i']['f3_var'], label=subject_name, color=colors[subject_name])
    ax.set_title('[i] - Variance')
    ax.set_xlabel('F2 variance (Hz)')
    ax.set_ylabel('F3 variance (Hz)')
    #ax.set_xlim(0,6000)
    #ax.set_ylim(0,6000)
    plt.legend()
    #fig.suptitle("Regular formant vs. formant normalized with P_min")
    print("random phones about to show!")
    plt.show()

    # plotting the ACS data
    fig, ax = plt.subplots()
    for subject_name in subj_list:
        if 'ɪ' not in clean_formant_dict[subject_name].keys():
            continue
        #import pdb; pdb.set_trace()
        if not ("subject" in subject_name or "CONTROL" in subject_name):
            print("didnt work")
            continue
        plt.scatter(clean_formant_dict[subject_name]['ɪ']['f2'], clean_formant_dict[subject_name]['ɪ']['f3'], label=subject_name, color=colors[subject_name])
    ax.set_title('[ɪ] - ACS Data')
    ax.set_xlabel('F2 value (Hz)')
    ax.set_ylabel('F3 value (Hz)')
    ax.set_xlim(0,6000)
    ax.set_ylim(0,6000)
    plt.legend()
    #fig.suptitle("Regular formant vs. formant normalized with P_min")
    print("random phones about to show!")
    plt.show()

    # plotting the SPA data, including control from ACS data
    fig, ax = plt.subplots()
    for subject_name in subj_list:
        if 'ɪ' not in clean_formant_dict[subject_name].keys():
            continue
        if not ("russian" in subject_name or "CONTROL" in subject_name):
            continue
        plt.scatter(clean_formant_dict[subject_name]['ɪ']['f2'], clean_formant_dict[subject_name]['ɪ']['f3'], label=subject_name, color=colors[subject_name])
    ax.set_title('[ɪ] - SPA Data')
    ax.set_xlabel('F2 value (Hz)')
    ax.set_ylabel('F3 value (Hz)')
    ax.set_xlim(0,6000)
    ax.set_ylim(0,6000)
    plt.legend()
    #fig.suptitle("Regular formant vs. formant normalized with P_min")
    print("random phones about to show!")
    plt.show()

    # plotting the F3 formants to find [l] velarization
    fig, ax = plt.subplots()
    subj_no = 1
    for subject_name in subj_list:
        subj_no += 1
        if not ('l' in clean_formant_dict[subject_name].keys() or 'ʟ' in clean_formant_dict[subject_name].keys()):
            continue
        list_of_l_f1 = clean_formant_dict[subject_name]['l']['f1']
        list_of_l_f2 = clean_formant_dict[subject_name]['l']['f2']
        list_of_l_f3 = clean_formant_dict[subject_name]['l']['f3']
        coord_test = subj_no
        if subject_name == "CONTROL":
            subj_no -= 1
            coord_test = 1
        if 'ʟ' in clean_formant_dict[subject_name].keys():
            print("made it here!")
            list_of_l_f2 = np.concatenate((list_of_l_f1, np.array(clean_formant_dict[subject_name]['ʟ']['f1'])))
            list_of_l_f2 = np.concatenate((list_of_l_f2, np.array(clean_formant_dict[subject_name]['ʟ']['f2'])))
            list_of_l_f3 = np.concatenate((list_of_l_f3, np.array(clean_formant_dict[subject_name]['ʟ']['f3'])))
        # pdb; pdb.set_trace()
        plt.scatter(list_of_l_f2, np.zeros_like(list_of_l_f2) + coord_test, label=subject_name, color=colors[subject_name])
        #plt.scatter(list_of_l_f2, list_of_l_f3, label=subject_name, color=colors[subject_name])
    
    ax.set_title('[l] velarization')
    ax.set_xlabel('F2 value (Hz)')
    ax.get_yaxis().set_visible(False)
    ax.set_ylabel('F3 value (Hz)')
    ax.set_xlim(0,4000)
    ax.set_ylim(0,10)
    plt.legend()
    #fig.suptitle("Regular formant vs. formant normalized with P_min")
    print("random phones about to show!")
    plt.show()

    # plotting the l2 against perceived data
    fig, ax = plt.subplots()
    name_list = ['CONTROL', 'subject_4', 'subject_7', 'subject_2', 'subject_1']
    ind = 0
    for subject_name in name_list:
        list_of_l_f2 = clean_formant_dict[subject_name]['l']['f2']
        if 'ʟ' in clean_formant_dict[subject_name].keys():
            print("made it here!")
        #    list_of_l_f2 = np.concatenate((list_of_l_f1, np.array(clean_formant_dict[subject_name]['ʟ']['f2'])))
        plt.scatter(perceived_list[ind], clean_formant_dict[subject_name]['l']['f2_avg'], label=subject_name, color=colors[subject_name])
        ind += 1
    ax.set_title('[l] vs. Perceived Strength')
    ax.set_xlabel('Perceived Accent Strength')
    ax.set_ylabel('F2 value (Hz)')
    ax.set_xlim(0,6000)
    ax.set_ylim(0,6000)
    plt.legend()
    #fig.suptitle("Regular formant vs. formant normalized with P_min")
    print("random phones about to show!")
    plt.show()

def clean_formants_please(csi, subj_list, phone_list):
    control_unclean = csi['CONTROL']
    control_clean = {}
    clean_formants = {}
    for subj in subj_list:
        clean_formants[subj] = {}
    for phone in phone_list:
        if phone not in control_unclean:
            continue
        control_clean[phone] = {}
        control_clean[phone]['f1'] = np.subtract(np.array(control_unclean[phone]['f1']), np.array(control_unclean[phone]['p_min']))
        control_clean[phone]['f2'] = np.subtract(np.array(control_unclean[phone]['f2']), np.array(control_unclean[phone]['p_min']))
        control_clean[phone]['f3'] = np.subtract(np.array(control_unclean[phone]['f3']), np.array(control_unclean[phone]['p_min']))
        control_clean[phone]['f1_avg'] = np.average(control_clean[phone]['f1'])
        control_clean[phone]['f2_avg'] = np.average(control_clean[phone]['f2'])
        control_clean[phone]['f3_avg'] = np.average(control_clean[phone]['f3'])
        for subj in csi:
            #import pdb; pdb.set_trace()
            if phone not in csi[subj]:
                continue
            clean_formants[subj][phone] = {}
            clean_formants[subj][phone]['f1_avg'] = np.average(np.subtract(np.array(csi[subj][phone]['f1']), np.array(csi[subj][phone]['p_min'])))
            clean_formants[subj][phone]['f1'] = np.subtract(np.array(csi[subj][phone]['f1']), np.array(csi[subj][phone]['p_min']))
            clean_formants[subj][phone]['f1_var'] = np.var(clean_formants[subj][phone]['f1'])
            clean_formants[subj][phone]['f1_avg_diff'] = np.subtract(clean_formants[subj][phone]['f1_avg'], control_clean[phone]['f1_avg'])
            clean_formants[subj][phone]['f2_avg'] = np.average(np.subtract(np.array(csi[subj][phone]['f2']), np.array(csi[subj][phone]['p_min'])))
            clean_formants[subj][phone]['f2'] = np.subtract(np.array(csi[subj][phone]['f2']), np.array(csi[subj][phone]['p_min']))
            clean_formants[subj][phone]['f2_var'] = np.var(clean_formants[subj][phone]['f2'])
            clean_formants[subj][phone]['f2_avg_diff'] = np.subtract(clean_formants[subj][phone]['f2_avg'], control_clean[phone]['f2_avg'])
            clean_formants[subj][phone]['f3_avg'] = np.average(np.subtract(np.array(csi[subj][phone]['f3']), np.array(csi[subj][phone]['p_min'])))
            clean_formants[subj][phone]['f3'] = np.subtract(np.array(csi[subj][phone]['f3']), np.array(csi[subj][phone]['p_min']))
            clean_formants[subj][phone]['f3_var'] = np.var(clean_formants[subj][phone]['f3'])
            clean_formants[subj][phone]['f3_avg_diff'] = np.subtract(clean_formants[subj][phone]['f3_avg'], control_clean[phone]['f3_avg'])
            clean_formants[subj][phone]['euclid'] = []
            for i in range(len(clean_formants[subj][phone]['f1'])):
                clean_formants[subj][phone]['euclid'].append(np.linalg.norm(np.array(clean_formants[subj][phone]['f1'][i], clean_formants[subj][phone]['f2'][i]) - np.array(control_clean[phone]['f1_avg'], control_clean[phone]['f2_avg'])))
            clean_formants[subj][phone]['euclid_avg'] = np.linalg.norm(np.array(clean_formants[subj][phone]['f1_avg'], clean_formants[subj][phone]['f2_avg']) - np.array(control_clean[phone]['f1_avg'], control_clean[phone]['f2_avg']))  # gives the Euclidean distance between this subject's phone and control
    import pdb; pdb.set_trace()
    return clean_formants
            

def print_euclids(clean_formants, subj_list, phone_list):
    phone_number = 0
    #vowels = ['ɛ', 'æ', 'u', 'o', 'i', 'ɪ', 'ʊ', 'a', 'eɪ', 'aɪ', 'ou', 'oʊ', 'ə']
    #if not vowels:
    vowels = ['i', 'ɪ', 'ɨ', 'o']
    for phone in phone_list:
        print("PHONE: ", phone)
        for subj in subj_list:
            if phone not in clean_formants[subj]:
                continue
            #if vowel_print and phone not in vowels:
            #    continue
            if phone not in vowels:
                continue
            print(subj, " ALL EUCLID DISTS: ", clean_formants[subj][phone]['euclid'])
            print(subj, " MEAN EUCLID DIST: ", clean_formants[subj][phone]['euclid_avg'])
            print(subj, " F1 MEAN: ", clean_formants[subj][phone]['f1_avg'], " VARIANCE: ", clean_formants[subj][phone]['f1_var'], " AVG DIFF: ", clean_formants[subj][phone]['f1_avg_diff'])
            print(subj, " F2 MEAN: ", clean_formants[subj][phone]['f2_avg'], " VARIANCE: ", clean_formants[subj][phone]['f2_var'], " AVG DIFF: ", clean_formants[subj][phone]['f2_avg_diff'])
            print(subj, " F3 MEAN: ", clean_formants[subj][phone]['f3_avg'], " VARIANCE: ", clean_formants[subj][phone]['f3_var'], " AVG DIFF: ", clean_formants[subj][phone]['f3_avg_diff'])


def plot_f0s(pitch_info, max_t_elapsed, max_t_elapsed_spa): # basically need to split up the SPA and hand-collected data into two subplots
    #plt.figure(2)
    fig, axs = plt.subplots(1, 2, sharey=True)
    print("THIS IS MAX_T_ELAPSED: ", max_t_elapsed, max_t_elapsed_spa)
    spa = False
    for subj in pitch_info:
        if "russian" in subj:
            spa = True
            scale_factor = max_t_elapsed_spa/pitch_info[subj]['total_dur'] # fix so that there is diff scaling factor for SPA and hand-collected

        else:
            spa = False
            scale_factor = max_t_elapsed/pitch_info[subj]['total_dur'] # fix so that there is diff scaling factor for SPA and hand-collected

        durs_scaled = [x / scale_factor for x in pitch_info[subj]['durs']]
        if spa:
            axs[0].plot(durs_scaled, pitch_info[subj]['pitches'], label=subj)
        else:
            axs[1].plot(durs_scaled, pitch_info[subj]['pitches'], label=subj)
    plt.legend()
    print("scaled pitch durs about to show!")
    plt.show()
    import pdb; pdb.set_trace()
    print("hello")


def plot_ash_e_perceived(perceived_list, formant_dict, colors):
    fig, ax = plt.subplots()
    name_list = ['CONTROL', 'subject_4', 'subject_7', 'subject_2', 'subject_1']
    ind = 0
    import pdb; pdb.set_trace()
    for subject_name in name_list:
        if 'ʊ' not in formant_dict[subject_name].keys():
            continue
        #import pdb; pdb.set_trace()
        if not ("subject" in subject_name or "CONTROL" in subject_name):
            print("didnt work")
            continue
        plt.scatter(perceived_list[ind], formant_dict[subject_name]['ʊ']['f2_avg'], label=subject_name, color=colors[subject_name])
        ind += 1
    ax.set_title('ʊ Normalized F2 vs. Perceived accent strength')
    ax.set_xlabel('Perceived accent strength')
    ax.set_ylabel('F2 (P_mean-normalized) (Hz)')
    #ax.set_xlim(0,5)
    #ax.set_ylim(0,6000)
    print_euclids(formant_dict, name_list, ['ʊ'])
    plt.legend()
    #fig.suptitle("Regular formant vs. formant normalized with P_min")
    print("random phones about to show!")
    plt.show()   


def consonant_perceived_strength(perceived_list):
    name_list = ['CONTROL', 'subject_4', 'subject_7', 'subject_2', 'subject_1']
    final_devoicing = [0, 0, 1, 0, 2]
    l_velar = [0, 0, 1, 4, 5]
    h_velar = [0, 0, 1, 1, 2]
    r_trilling = [0, 1, 2, 4, 5]
    vowel_pal = [0, 5, 4, 7, 5]
    fig, ax = plt.subplots()
    ind = 0
    #for subject_name in name_list:
    #    plt.scatter(perceived_list[ind], final_devoicing[ind], label=subject_name)
    #    ind += 1
    plt.scatter(perceived_list, final_devoicing, label='Final devoicing')
    plt.scatter(perceived_list, l_velar, label='Liquid velarization')
    plt.scatter(perceived_list, h_velar, label='[h] velarization')
    plt.scatter(perceived_list, r_trilling, label='[r] trilling')
    plt.scatter(perceived_list, vowel_pal, label='Vowel palatalization')
    plt.plot(perceived_list, final_devoicing)
    plt.plot(perceived_list, l_velar)
    plt.plot(perceived_list, h_velar)
    plt.plot(perceived_list, r_trilling)
    plt.plot(perceived_list, vowel_pal)

    ax.set_title('Consonant trends vs. Perceived accent strength')
    ax.set_xlabel('Perceived accent strength')
    ax.set_ylabel('# Occurrences')
    ax.set_ylim(0, 8)
    plt.legend(loc= 'upper right', prop={'size': 6})
    plt.show()

if __name__ == "__main__":
    general_function(sys.argv[1])

