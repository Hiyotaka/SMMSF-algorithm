function [MST_1,MST_3]=kmst(data)
dis = pdist2(data, data);
G = graph(dis);%计算完全图
allEdges = [];
%循环建立最小生成树
for i = 1:3
    [T, ~] = minspantree(G, "Method", "dense");
    if i==1
        MST_1=T;
    end
    edges=T.Edges.EndNodes;
    allEdges = [allEdges; edges];
    G = rmedge(G, edges(:,1), edges(:,2));
end
%转邻接矩阵
MST_1=full(adjacency(MST_1)).*dis;
MST_3 = dis .* full(adjacency(graph(allEdges(:,1), allEdges(:,2))));
end