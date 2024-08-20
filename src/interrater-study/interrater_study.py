import geojson
import os
import pandas as pd
import cv2
import numpy as np
from sklearn.metrics import cohen_kappa_score
from glob import glob
import argparse
from functools import reduce

def json_load(json_dir,geofile1):
    with open(os.path.join(json_dir,geofile1)) as f:
        gj = geojson.load(f)
    features = gj['features']
    return features


def polygon_rater_data_load(json_dir,geofile1):
    features1 = json_load(json_dir,geofile1)
    feature1_points = [f for f in features1 if (f["geometry"]["type"]=="Polygon") & ("isLocked" not in f["properties"].keys())]
    feature1_coords = [cords["geometry"]["coordinates"] for cords in feature1_points]
    feature1_name = []
    feature1_class = []
    for cords in feature1_points:
        if "name" not in cords["properties"].keys():
            feature1_name.append(None)
        else:
            feature1_name.append(cords["properties"]["name"])
        if "classification" not in cords["properties"].keys():
            feature1_class.append(None)
        else:
            feature1_class.append(cords["properties"]["classification"]["name"])
    feature1_df = pd.DataFrame({"coordinates":feature1_coords,"class":feature1_class, "name":feature1_name})
    return feature1_coords, feature1_class, feature1_name, feature1_df


def point_rater_data_load(json_dir,geofile2):
    features2 = json_load(json_dir,geofile2)
    features2_points = [f for f in features2 if f["geometry"]["type"]=="Point"]
    feature2_coords = [cords["geometry"]["coordinates"] for cords in features2_points]
    feature2_class = [cords["properties"]["classification"]["name"] if "classification" in (cords["properties"].keys()) else "None" for cords in features2_points ]
    feature2_name = [cords["properties"]["name"] if "name" in (cords["properties"].keys()) else "None" for cords in features2_points]
    feature2_df = pd.DataFrame({"coordinates":feature2_coords,"class":feature2_class, "name":feature2_name})
    return feature2_coords, feature2_class, feature2_name, feature2_df



def match_main_point_rater(feature1_coords,feature1_class,feature1_name,feature2_coords,feature2_class,feature2_name, radius, rater2_type):
    match = []
    i=0
    for c, class_var, name_var in zip(feature1_coords,feature1_class,feature1_name):
        #print(contours,class_var,name_var)
        j=0
        for cnt, class_var1, name1 in zip(feature2_coords,feature2_class,feature2_name):
            #print(c, class_var1, name1)
            #print((int(c[0]),int(c[1])))
            
            contours = [[int(c[0])+radius,int(c[1])],[int(c[0])+0.5*radius,int(c[1])+0.5*radius], [int(c[0]),int(c[1])+radius], [int(c[0])-0.5*radius,int(c[1])+0.5*radius], [int(c[0])-radius,int(c[1])], 
                        [int(c[0])-0.5*radius,int(c[1])-0.5*radius], [int(c[0])-radius,int(c[1])],[int(c[0])+0.5*radius,int(c[1])-0.5*radius]]
            
            if (rater2_type=="Polygon"):
                #print("cnt[0])>2")
                cnt = np.mean(cnt[0], axis=0)
            #print(len(cnt))
            dist = cv2.pointPolygonTest(np.int32(np.array(contours).round()),(int(cnt[0]),int(cnt[1])),False)
            if dist>=1:
                match.append([i,class_var,name_var,j,class_var1,name1, contours, cnt])
            j=j+1
        i=i+1 
    df = pd.DataFrame(match, columns=["main_rater_index","main_rater_class","main_rater_object_name","rater1_index","rater1_class","rater1_object_name","polygon_coords","point_coords"])
    return df


def match_point_rater(feature1_coords,feature1_class,feature1_name,feature2_coords,feature2_class,feature2_name):
    match = []
    i=0
    for contours, class_var, name_var in zip(feature1_coords,feature1_class,feature1_name):
        #print(contours,class_var,name_var)
        j=0
        for c, class_var1, name1 in zip(feature2_coords,feature2_class,feature2_name):
            #print(c, class_var1, name1)
            #print((int(c[0]),int(c[1])))
            dist = cv2.pointPolygonTest(np.int32(np.array(contours).round()),(int(c[0]),int(c[1])),False)
            if dist>=1:
                match.append([i,class_var,name_var,j,class_var1,name1, contours, c])
            j=j+1
        i=i+1 
    df = pd.DataFrame(match, columns=["main_rater_index","main_rater_class","main_rater_object_name","rater1_index","rater1_class","rater1_object_name","polygon_coords","point_coords"])
    return df


def compute_kappa_score(df, rater1_column, rater2_column):
    #df1= df[~((df["main_rater_class"]=='None') & (df["main_rater_class"]=='None'))]
    labeler1 = df[rater1_column]
    labeler2 = df[rater2_column]
    return cohen_kappa_score(labeler1, labeler2)



def find_missing_main_rater(df, feature1_coords,feature1_class,feature1_name):
    missing_main_rater_list = []
    for i in range(len(feature1_coords)):
        if i not in df["main_rater_index"].values:
            #print(i,feature1_name[i], feature1_coords[i])
            missing_main_rater_list.append([i,feature1_class[i],feature1_name[i],None, None,None, feature1_coords[i],None])
    missing_main_rater = pd.DataFrame(missing_main_rater_list, columns=["main_rater_index","main_rater_class","main_rater_object_name","rater1_index","rater1_class","rater1_object_name","polygon_coords","point_coords"])
    return missing_main_rater


def find_missing_rater1(df, feature2_coords,feature2_class,feature2_name):
    missing_rater1_list = []
    for i in range(len(feature2_class)):
        if i not in df["rater1_index"].values:
            #print(i,feature2_class[i],feature2_coords[i] )
            missing_rater1_list.append([None, None,None,i,feature2_class[i],feature2_name[i], None, feature2_coords[i]])
    missing_rater1 = pd.DataFrame(missing_rater1_list,columns=["main_rater_index","main_rater_class","main_rater_object_name","rater1_index","rater1_class","rater1_object_name","polygon_coords","point_coords"])
    return missing_rater1


def find_match(json_dir1,json_dir2, geojsons_names,radius,main_rater_type, rater2_type):
    all_geojson_df = pd.DataFrame()
    for geofile1 in geojsons_names:
        #print("------------------",geofile1,"---------------------")
        
        if rater2_type=="Polygon":
            feature2_coords, feature2_class, feature2_name, feature2_df = polygon_rater_data_load(json_dir2,geofile1)
        if rater2_type=="Point":
            feature2_coords, feature2_class, feature2_name, feature2_df = point_rater_data_load(json_dir2,geofile1)
            
            
        if main_rater_type=="Point":
            feature1_coords, feature1_class, feature1_name, feature1_df = point_rater_data_load(json_dir1,geofile1)
            df = match_main_point_rater(feature1_coords,feature1_class,feature1_name,feature2_coords,feature2_class,feature2_name, radius,rater2_type)
            
        if main_rater_type=="Polygon": 
            feature1_coords, feature1_class, feature1_name, feature1_df = polygon_rater_data_load(json_dir1,geofile1)
            df = match_point_rater(feature1_coords,feature1_class,feature1_name,feature2_coords,feature2_class,feature2_name)
        #print("---", df.head(2))
        missing_main_rater = find_missing_main_rater(df, feature1_coords,feature1_class,feature1_name)
        #missing_rater1 = find_missing_rater1(df, feature2_coords,feature2_class,feature2_name)
        #df_final = pd.concat([df,missing_rater1,missing_main_rater], axis=0, ignore_index=True) 
        df_final = pd.concat([df,missing_main_rater], axis=0, ignore_index=True) 
        df_final["main_rater_class"] = np.where(df_final["main_rater_class"].isna(),"None",df_final["main_rater_class"])
        df_final["rater1_class"] = np.where(df_final["rater1_class"].isna(),"None",df_final["rater1_class"])
        df_final["geojson_file"] = geofile1
        if len(all_geojson_df)==0:
            all_geojson_df = df_final
        else:
            all_geojson_df =  pd.concat([all_geojson_df,df_final], ignore_index=True)
    return all_geojson_df

def filter_noncommon_rows(df, columns_to_check):
    # Generalized condition: select rows where any of the specified columns have the value "None"
    condition = reduce(lambda x, y: x | y, [(df[col] == "None") for col in columns_to_check])
    return df[condition]

def fleiss_kappa1(lists, classes):
    n = len(lists)
    N = len(lists[0])
    k = len(classes)
    print("n N k", n, N, k)
    nij = []
    for i in range(N):
        nij.append([0]*k)
        
    
    for i in range(len(lists)):
        for j in range(len(lists[i])):
            nij[j][classes.index(lists[i][j])] += 1 
    
    P = []
    for i in nij:
        P.append(1/(n*(n-1))*(sum([j*j for j in i])-n))
    return (((sum(P)/N)-(sum([y*y for y in [x/(N*n) for x in[sum(i) for i in zip(*nij)]]])))/(1-sum([y*y for y in [x/(N*n) for x in[sum(i) for i in zip(*nij)]]]))+1)/2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file_path")
    parser.add_argument("save_dir_path")
    args = parser.parse_args()
    filepaths = args.csv_file_path
    save_dir = args.save_dir_path
    csv_file = pd.read_csv(filepaths)
    radius = 200

    ggeojsons_main_rater =  glob(os.path.join(csv_file["Geojson_path"].iloc[0],"*.geojson"))

    geojsons_names =  [x.split("/")[-1] for x in ggeojsons_main_rater]
    
    #match_all_raters = []
    rater_names = csv_file["Rater_Name"].values
    
    print("geojsons_names", geojsons_names)
    print("rater_names", rater_names)
    main_rater_df_dict={}
    
    for i in range(len(csv_file)):
        tmp = []
        matched_df= pd.DataFrame()
        for j in range(len(csv_file)):
            if i!=j:
                df = find_match(csv_file["Geojson_path"].iloc[i],csv_file["Geojson_path"].iloc[j], geojsons_names,radius,csv_file["Rater_Type"].iloc[i], csv_file["Rater_Type"].iloc[j])
                df = df[['main_rater_index', 'main_rater_class', 'main_rater_object_name','rater1_index', 'rater1_class', 'rater1_object_name','geojson_file']]
                df.columns = [csv_file["Rater_Name"].iloc[i]+'_index', csv_file["Rater_Name"].iloc[i]+'_class', csv_file["Rater_Name"].iloc[i]+'_object_name', csv_file["Rater_Name"].iloc[j]+'_index', csv_file["Rater_Name"].iloc[j]+'_class', csv_file["Rater_Name"].iloc[j]+'_object_name','geojson_file']
                if len(matched_df)>0:
                    matched_df = pd.merge(matched_df, df, on=[csv_file["Rater_Name"].iloc[i]+'_index', csv_file["Rater_Name"].iloc[i]+'_class', csv_file["Rater_Name"].iloc[i]+'_object_name','geojson_file'],how='left')
                else:
                    matched_df= df
        main_rater_df_dict[csv_file["Rater_Name"].iloc[i]] =matched_df
    #print(main_rater_df_dict.keys())  
    

    
    columns_to_check = [x+"_class" for x in rater_names]
    
    raters_df = list(main_rater_df_dict.keys())
    
    common = main_rater_df_dict[raters_df[0]][reduce(lambda x, y: x & y, [(main_rater_df_dict[raters_df[0]][col] != "None") for col in columns_to_check])]
    noncommon_list =[common]
    
    for k,v in main_rater_df_dict.items():
        columns_to_check_2  = [x+"_class" for x in rater_names]
        columns_to_check_2.remove(k+"_class")
        noncommon = filter_noncommon_rows(v, columns_to_check_2)
        noncommon_list.append(noncommon)
        

    final_output = pd.concat(noncommon_list).drop_duplicates()
    
    final_output.to_csv(os.path.join(save_dir,"all_objects_full.csv"))
    
    all_objects = final_output[columns_to_check]
    

    
    for col in columns_to_check:
        all_objects[col] = np.where(all_objects[col]=="None","",all_objects[col])
        
    all_objects= all_objects.reset_index()
    all_objects["object"]=all_objects.index
    all_objects.drop(["index"],axis=1,inplace=True)
    
    all_objects.to_csv(os.path.join(save_dir, "all_objects.csv"))
    
    pairwise_kappa = []
    rater1 = []
    rater2=[]
    
    for i in range(len(columns_to_check)):
        for j in range(i+1, len(columns_to_check)):
            k1 = compute_kappa_score(final_output, columns_to_check[i],columns_to_check[j])
            rater1.append(columns_to_check[i].split("_")[0])
            rater2.append(columns_to_check[j].split("_")[0])
            pairwise_kappa.append(k1)
    cohens_kappa = pd.DataFrame({"Rater 1":rater1, "Rater 2":rater2, "cohens_kappa":pairwise_kappa })    
    #cohens_kappa = pd.DataFrame(pairwise_kappa, columns=["Rater 1","Rater 2","Cohen Kappa Score"])
    cohens_kappa.to_csv(os.path.join(save_dir,"cohens_kappa.csv"))
    print("cohens_kappa", cohens_kappa)

    dict1 = {}
    max_class_list = []
    for k in range(len(columns_to_check)):
        class_list_check = all_objects[columns_to_check[k]].unique()
        max_class_list.extend(class_list_check)
        
    classes  = set(max_class_list)
    print("classes found", classes)
    
    for idx, class_val in enumerate(classes):
        if len(class_val)==0:
            dict1["None"]=idx+1
        else:
            dict1[class_val]=idx+1
    #dict1 = {"Coarse-Grained":1, "Diffuse":2, "Cored":3, "None":4}
    print(dict1)
    
    table = final_output[columns_to_check]
    
    table= table.fillna("None")
    #print(len(table))
    
    for col in columns_to_check:
        table[col] = table[col].apply(lambda l: dict1[l])
    
    
    l=[]
    for col in columns_to_check:
        l.append(table[col].values)
        
    fleiss_kappa = fleiss_kappa1(l,list(np.arange(1,len(classes)+1)))
    pd.DataFrame({"fleiss_kappa":[fleiss_kappa]}).to_csv(os.path.join(save_dir,"fleiss_kappa.csv"))
    print("fleiss_kappa", fleiss_kappa)
    
    