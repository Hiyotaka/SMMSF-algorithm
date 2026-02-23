function [IDX]=SMMSF_clustering(data,K)
%%
%Normalization
data=(data-min(data))./(max(data)-min(data));
data(isnan(data))=0;
%%
%Construct k-MST
[MST_1,MST_3]=kmst(data);
%Split
[clusters,clustersCenters]=splitTree(data,MST_1,MST_3);
%Merge
IDX=TwoRoundMerge(clusters,MST_1,MST_3,data,K,clustersCenters);
end