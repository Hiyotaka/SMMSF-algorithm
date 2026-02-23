function [clusters,clustersCenter]=splitTree(data,MST_1,MST_3)
N=size(data,1);
clusters={};
clustersCenter=[];
num_clu=0;
t1=MST_1;
t1(t1>0)=1;
% 计算每个节点的平均权重
averageWeights_T1 = sum(MST_1, 2) ./ sum(t1, 2);
MST1Degree=sum(t1, 2);
rho1=MST1Degree./averageWeights_T1;
%%
%计算MST_3中每个点的密度
% 计算每个节点的平均权重
averageWeights_T1_T2_T3 = sum(MST_3, 2) ./ sum(MST_3>0, 2);
for i = 1:size(MST_3, 1)
    for j = 1:size(MST_3, 2)
        if MST_3(i, j) > 2 * averageWeights_T1_T2_T3(i)
            MST_3(i, j) = 0;
            MST_3(j, i) = 0; % 由于MST_3是对称的，需要同时修改对称位置的值
        end
    end
end%对于其中较长的边对平均权重影响大的断开
t3 = MST_3>0;
MST3Degree=sum(t3, 2);
averageWeights_T1_T2_T3 = sum(MST_3, 2) ./ MST3Degree;
%再次计算平均权重
rho3=MST3Degree./averageWeights_T1_T2_T3;
rho=rho3-rho1;%密度变化率
if exist('makehotmap','file') == 2
    makehotmap(data,rho);
end
%%
connectedpts={};
Rep=zeros(N,1);
for i=1:N
    connectedpts{i,1}=[i,find(MST_3(i,:)>0)];
    Rep(i,1)=findRepresent(rho(connectedpts{i,1},1),connectedpts{i,1});
end%Rep中存储每个点的代表点
visited=zeros(N,1);
flag=0;
for i=1:N
    if visited(i,1)==0 
        sup=i;         
        flag=flag+1;
        while Rep(sup,1)~=sup
            visited(sup,1)=flag;
            sup=Rep(sup,1);
        end
    end
    for j=1:N
        if visited(j,1)==flag
            Rep(j,1)=sup;
        end
    end
end
for i=1:N
    if Rep(i,1)==i
        num_clu=num_clu+1;
        clustersCenter=[clustersCenter;i];
    end
end
for i=1:num_clu
    cc=find(Rep==clustersCenter(i,1));
    clusters{i,1}=cc;
end
%单个点就近分配至最近的小簇
[clusters,clustersCenter] = redistributeSingletons(data, clusters,clustersCenter);
end
%%
%寻找代表点
function index=findRepresent(rowRho,rowConnected)
[~,MaxIndex]=max(rowRho);
index=rowConnected(MaxIndex);
end
%%
%分配单个数据点
function [clusters,clustersCenter] = redistributeSingletons(data, clusters,clustersCenter)
clusterSizes = cellfun(@numel, clusters);
hasSingleton = any(clusterSizes == 1);
if ~hasSingleton
    return;
end
numClusters = numel(clusters);
removeIdx = false(numClusters, 1);
for i = 1:numClusters
    if length(clusters{i,1}) == 1
        singletonPoint = data(clusters{i,1}, :);
        minDistance = inf;
        nearestCluster = 0;
        for j = 1:numClusters
            if i ~= j
                distances = pdist2(singletonPoint, data(clusters{j,1}, :));
                [minDistToCluster, ~] = min(distances);
                if minDistToCluster < minDistance
                    minDistance = minDistToCluster;
                    nearestCluster = j;
                end
            end
        end
        clusters{nearestCluster} = [clusters{nearestCluster,1}; clusters{i,1}];
        clusters{i,1} = [];
        removeIdx(i) = true;
    end
end
clusters = clusters(~cellfun('isempty', clusters));
clustersCenter = clustersCenter(~removeIdx, :);
end
