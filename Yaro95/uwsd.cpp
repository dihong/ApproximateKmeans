/*
 * This program implements the word sense disambiguation algorithm in:
 * 	D. Yarowsky, "Unsupervised Word Sense Disambiguation Rivaling Supervised Methods", 1995.
 * How to use:
 *	1) train a classifier for word to be disambiguated: uwsd -train /path/to/training/text /path/to/seeds /path/to/classifer word-to-be-disambiguated
 *	2) Disambiguate the same word for new sentenses:    uwsd -test /path/to/classifer /path/to/ambtext word-to-be-disambiguated
 * Description:
 *	/path/to/training/text:     A path of a training file that contains training text for word to be disambiguated, one document (e.g. tweet) per line.
 *	/path/to/seeds:		    	A path of a seed file. The number of lines in this file is equal to the number of different senses for the word.
 *				    			Each line contains one or more keywords that ACCURATELY describe the sense at that line.
 * 	/path/to/classifer:         A path specifies where the classifer (decision rules) will be saved.
 *	word-to-be-disambiguated:	The word to be disambiguated, e.g. tank or bank.
 * 	/path/to/ambtext:   		A path to text to be disambiguated.
 *
 *
 * Author: Dihong Gong
 * Date:  Jan 14, 2015
 */

#include<stdio.h>
#include<string>
#include<string.h>
#include<vector>
#include "libstemmer.h"
#include<map>
#include<algorithm>

using namespace std;

#define winsize 10 //window size around ambiguous word, at least 3.
#define score_threshold 0.4

pthread_mutex_t mutex_scenes = PTHREAD_MUTEX_INITIALIZER;

struct sb_stemmer * stemmer = 0; //stemming

typedef struct DECLIST{  //decision list structs.
    vector<double> conf;  //confidence conf, higher is better. Calculated as: (max_count+1)/(second_max_count+1).
    vector<int> pos; // -1/+1, winsize
    vector<string> determinants;  //words used to determine senses.
    vector<int> prediction;
}DECLIST;


typedef struct INDEX{
    vector<int> text_id;  //the id of text containing the word. starting at 0.
    vector<int> pos;  //position of this word in the corresponding text. -winsize:winsize
}INDEX;

vector<string> stem(const char* keywords){ //"There are 3 students" => "THERE ARE 3 STUDENT". The input must ends with '\0'
    char temp[256];
    int len = strlen(keywords);
    if(len>sizeof(temp)-1)
        len = sizeof(temp)-1;
    memcpy(temp,keywords,len);
    temp[len] = 0;
    int begin = 0;
    int end = 0;
    vector<string> ret;
    string tmp;
    while (temp[begin]){
        if (temp[begin]>64 && temp[begin]<91)
            temp[begin] += 32;
        else if ((temp[begin]>122 || temp[begin]<97) && !(temp[begin]>='0' && temp[begin]<='9'))
            temp[begin] = ' ';
        begin++;
    }
    begin = 0;
    end = 0;
    while (temp[begin]) {
        while (temp[end] != 0 && temp[end] != ' ') end++;
        tmp = (char*) sb_stemmer_stem(stemmer, (sb_symbol*) (temp + begin), end - begin);
        ret.push_back(tmp);
        if (temp[end] == 0) break;
        begin = end + 1;
        //skip multiple blanks.
        while(temp[begin] != 0 && temp[begin] == ' ') begin++;
        end = begin;
    }
    return ret;
}

bool mycompfunc_double(const pair<double, int>& l, const pair<double, int>& r) {
    return l.first > r.first;
}

void quick_sort(double* arr, int N, int* order, double* sorted, bool descend) {
    vector< pair<double, int> > WI;
    pair<double, int> val_ind;
    for (int i = 0; i < N; i++) {
        val_ind.first = arr[i]; //value.
        val_ind.second = i; //index.
        WI.push_back(val_ind);
    }
    sort(WI.begin(), WI.end(), mycompfunc_double);
    if (descend)
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        }
    else
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[N-i-1].first;
            order[i] = WI[N-i-1].second;
        }
}


void learn_rules(const int* label, DECLIST& decisionList, map<string,INDEX*>& idx, const int& num_scenes,string ambword){
    //learning rules based on current text labeling.
    int i,j,k;
    map<string,INDEX*>::iterator it;
    double* conf = new double [idx.size()];
    int* prediction = new int [idx.size()];
    int* dist = new int [num_scenes];
    int max1,max2,ind1,ind2;
    int cnt = 0;
    for(it = idx.begin();it!=idx.end();it++){
        //calculate the confidence of rules
        memset(dist,0,sizeof(int)*num_scenes);
        for(i=0;i<it->second->text_id.size();i++){
            if(label[it->second->text_id[i]]>0)  //this has been tagged.
                dist[label[it->second->text_id[i]]-1]++;
            //printf("tid: %d, pos: %d, word: %s.\n",1+it->second->text_id[i],it->second->pos[0],it->first.c_str());
        }
        max1 = dist[0]; ind1 = 0;
        for(i=1;i<num_scenes;i++){
            if(dist[i]>max1){
                max1 = dist[i];
                ind1 = i;
            }
        }
        if(ind1==0) ind2 = 1;
        else ind2 = 0;
        max2 = dist[ind2];
        for(i=0;i<num_scenes;i++){
            if(dist[i]>max2 && i!=ind1){
                max2 = dist[i];
                ind2 = i;
            }
        }
        conf[cnt] = 1 - (max2+1.0)/(max1+1.0);
        //if(conf[cnt]>0) printf("conf = %.4f, pred = %d, w = %s\n",conf[cnt],1+ind1,it->first.substr(0,it->first.size()-2).c_str());
        prediction[cnt] = 1+ind1;
        cnt++;
    }
    
    int* order = new int [idx.size()];
    
    quick_sort(conf, idx.size(), order, 0, true);
    vector<int>pos;
    vector<string> det;
    for(it = idx.begin();it!=idx.end();it++){
        pos.push_back(it->second->pos[0]);
        det.push_back(it->first.substr(0,it->first.size()-2));
    }
    DECLIST decisionList2;
    for(i=0;i<idx.size();i++){
        if(conf[order[i]]>score_threshold && det[order[i]]!=ambword){
            decisionList2.conf.push_back(conf[order[i]]);
            decisionList2.pos.push_back(pos[order[i]]);
            decisionList2.determinants.push_back(det[order[i]]);
            decisionList2.prediction.push_back(prediction[order[i]]);
            //printf("conf = %.4f, pred = %d, w = %s\n",conf[order[i]],prediction[order[i]],det[order[i]].c_str());
        }
    }
    
    decisionList = decisionList2;
    delete order,dist,conf;
}


void disambiguate_text(const vector< vector<string> >& text, int* label, const DECLIST& decisionList, double* confidence, int* pos_of_ambword){
    //Update label based on provided decisionList. Return label: 0 is undetermined. Positive means scenes labeling. Only update label of instances with stronger confidence.
    int i,j,k,l,m;
    for(i=0;i<text.size();i++){ //for each text instance.
        for(j=0;j<decisionList.pos.size();j++){ //test each rule.
            if(decisionList.pos[j]==-1){ //left immediate
                if(pos_of_ambword[i]>0 && text[i][pos_of_ambword[i]-1]==decisionList.determinants[j]){
                    if(decisionList.conf[j]>confidence[i]){
                        label[i] = decisionList.prediction[j];
                        confidence[i] = decisionList.conf[j];
                    }
                    break;
                }
            }
            else if(decisionList.pos[j]==1){  //right immediate.
                if(pos_of_ambword[i]<text[i].size()-1 && text[i][pos_of_ambword[i]+1]==decisionList.determinants[j]){
                    if(decisionList.conf[j]>confidence[i]){
                        label[i] = decisionList.prediction[j];
                        confidence[i] = decisionList.conf[j];
                    }
                    break;
                }
            }
            else{  //winsize.
                
                k = pos_of_ambword[i] - decisionList.pos[j];
                l = pos_of_ambword[i] + decisionList.pos[j];
                if(k<0) k = 0;
                if(l>text[i].size()) l = text[i].size();
                for(m=k;m<l;m++){
                    if(text[i][m]==decisionList.determinants[j]){
                        if(decisionList.conf[j]>confidence[i]){
                            label[i] = decisionList.prediction[j];
                            confidence[i] = decisionList.conf[j];
                        };
                        //printf("tid: %d, classified as: %d, by: %s\n",i+1,label[i],decisionList.determinants[j].c_str());
                        goto tag;
                    }
                }
            }
        }
        tag: ;
    }
}

void init_disambiguation(const vector< vector<string> >& text, int* label, const vector< vector<string> >& seeds,double*conf){
    //Generate initial label based on whether a text contains seed(s) or not. Return label: 0 is undetermined. Positive means initial sences labeling.
    int i,j,k,l;
    memset(label,0,sizeof(int)*text.size());
    memset(conf,0,sizeof(double)*text.size());
    for(i=0;i<text.size();i++){
        for(j=0;j<text[i].size();j++){
            for(k=0;k<seeds.size();k++){
                for(l=0;l<seeds[k].size();l++){
                    if(text[i][j]==seeds[k][l]){
                        label[i] = k+1;
                        conf[i] = 1;
                        goto tag;
                    }
                }
            }
        }
        tag:
            ;
    }
}

void save_decisionList(FILE* fp, const DECLIST& decisionList){
    int i;
    for(i=0;i<decisionList.conf.size();i++){
        fprintf(fp,"%d %.4f %d %s\n",decisionList.prediction[i],decisionList.conf[i],decisionList.pos[i],decisionList.determinants[i].c_str());
    }
}

void load_decisionList(FILE* fp, DECLIST& decisionList,char*buffer,int max_len_line){
    int pred,pos;
    float conf;
    char word[1000];
    while (!feof(fp)) {
        if (fgets(buffer, max_len_line, fp) != NULL) {
            if(sscanf(buffer,"%d %f %d %s\n",&pred,&conf,&pos,word)!=EOF){
                decisionList.prediction.push_back(pred);
                decisionList.conf.push_back(conf);
                decisionList.pos.push_back(pos);
                decisionList.determinants.push_back(word);
            }
        }
        else
            break;
    }
}

const int max_len_line = 100000;
char buffer[max_len_line]; //input buffer: maximun number of char of each line in any input file.

int i,j,k;

int main(int argc, char** argv){
    if(argc<4){
        puts("At least 3 parameters required.");
        return -1;
    }
    if(strcmp("-train",argv[1])==0){
        if(argc!=6){
            puts("Incorrect number of parameters for training mode.");
            return -1;
        }else{ /*train*/
            /*Initialize stemming*/
            stemmer = sb_stemmer_new("english", 0);
            if (!stemmer) {
                puts("Cannot initialize stemmer.");
                return -1;
            }
            //read inputs.
            FILE* fp_train_text = fopen(argv[2],"r");
            FILE* fp_seed = fopen(argv[3],"r");
            FILE* fp_out = fopen(argv[4],"w+");
            string word = stem(argv[5])[0];
            if(!fp_train_text){
                printf("Cannot open file '%s' for reading.\n",argv[2]);
                return -1;
            }
            if(!fp_seed){
                printf("Cannot open file '%s' for reading.\n",argv[3]);
                return -1;
            }
            if(!fp_out){
                printf("Cannot open file '%s' for writing.\n",argv[4]);
                return -1;
            }
            //read & stem seeds.
            vector< vector<string> > seeds;
            while (!feof(fp_seed)) {
                if (fgets(buffer, max_len_line, fp_seed) != NULL) {
                    seeds.push_back(stem(buffer));
                } else
                    break;
            }
            if(seeds.size()<2){
                printf("Error: At least two senses need to be specified in the seeds file.\n");
            }
            //read & stem text.
            vector< vector<string> > text;
            while (!feof(fp_train_text)) {
                if (fgets(buffer, max_len_line, fp_train_text) != NULL) {
                    text.push_back(stem(buffer));
                } else
                    break;
            }
            int* label1 = new int [text.size()];
            int* label2 = new int [text.size()];
            double* confidence = new double [text.size()];
            int* pos = new int [text.size()];
            map<string,INDEX*> idx;
            DECLIST decisionList;
            //Find the position of words to be disambiguated in each text sentences.
            for(i=0;i<text.size();i++){
                for(j=0;j<text[i].size();j++){
                    if(text[i][j]==word){
                        break;
                    }
                }
                pos[i] = j;
            }
            //calculate the index: all the words falling inside winsize.
            for(i=0;i<text.size();i++){
                if(pos[i]==text[i].size())
                    continue; //no ambiguous word contained here.
                for(j=0;j<text[i].size();j++){
                    if(j!=pos[i] && j-pos[i]>=-winsize && j-pos[i]<=winsize){  //word inside window.
                        if(j-pos[i]==-1){
                            if(idx.find(text[i][j]+"-1")==idx.end()) idx[text[i][j]+"-1"] = new INDEX;
                            idx[text[i][j]+"-1"]->text_id.push_back(i);
                            idx[text[i][j]+"-1"]->pos.push_back(j-pos[i]);
                        }
                        else if(j-pos[i]==1){
                            if(idx.find(text[i][j]+"+1")==idx.end()) idx[text[i][j]+"+1"] = new INDEX;
                            idx[text[i][j]+"+1"]->text_id.push_back(i);
                            idx[text[i][j]+"+1"]->pos.push_back(j-pos[i]);
                        }
                        else{
                            if(idx.find(text[i][j]+"=k")==idx.end()) idx[text[i][j]+"=k"] = new INDEX;
                            idx[text[i][j]+"=k"]->text_id.push_back(i);
                            idx[text[i][j]+"=k"]->pos.push_back(10);
                        }
                    }
                }
            }
            //Generate initial labeling based on given seeds.
            init_disambiguation(text, label1, seeds,confidence);
            bool converged = false;
            memcpy(label2,label1,sizeof(int)*text.size());  //initialize label2 as label 1.
            while(!converged){
                learn_rules(label1, decisionList, idx, seeds.size(),word);
                disambiguate_text(text, label2, decisionList, confidence,pos);
                converged = true;
                for(i=0;i<text.size();i++){
                    if(label1[i]!=label2[i]){
                        converged = false;
                        break;
                    }
                }
                memcpy(label1,label2,sizeof(int)*text.size());
            }
            save_decisionList(fp_out,decisionList);
            fclose(fp_train_text);
            fclose(fp_seed);
            fclose(fp_out);
        }
    }else if(strcmp("-test",argv[1])==0){
        if(argc!=5){
            puts("Incorrect number of parameters for training mode.");
            return -1;
        }else{
            /*test*/
            FILE* fp_classifer = fopen(argv[2],"r");
            FILE* fp_text = fopen(argv[3],"r");
            if(!fp_classifer){
                printf("Cannot open file '%s' for reading.\n",argv[2]);
                return -1;
            }
            if(!fp_text){
                printf("Cannot open file '%s' for reading.\n",argv[3]);
                return -1;
            }
            //Initialize stemming
            stemmer = sb_stemmer_new("english", 0);
            if (!stemmer) {
                puts("Cannot initialize stemmer.");
                return -1;
            }
            //read text to be disambiguated, one line per document.
            vector< vector<string> > text;
            while (!feof(fp_text)) {
                if (fgets(buffer, max_len_line, fp_text) != NULL) {
                    text.push_back(stem(buffer));
                } else
                    break;
            }
            //Find the position of words to be disambiguated in each text sentences.
            int* pos = new int [text.size()];
            double* confidence = new double [text.size()];
            int* label = new int [text.size()];
            string word = stem(argv[4])[0];
            for(i=0;i<text.size();i++){
                for(j=0;j<text[i].size();j++){
                    if(text[i][j]==word){
                        break;
                    }
                }
                pos[i] = j;
                if(j==text[i].size()){
                    confidence[i] = 1.0;
                    label[i] = -1; //ambiguous word not present.
                }
                else{
                    confidence[i] = 0;
                    label[i] = 0;
                }
            }
            DECLIST decisionList;
            load_decisionList(fp_classifer, decisionList, buffer, max_len_line);
            disambiguate_text(text, label, decisionList, confidence, pos);
            fclose(fp_classifer);
            fclose(fp_text);
            for(i=0;i<text.size();i++)
                printf("%d %.4f\n",label[i],confidence[i]);
        }
    }
    else{
        puts("Incorrect parameters: the first parameter must be either '-train' or '-test'.");
        return -1;
    }
}
