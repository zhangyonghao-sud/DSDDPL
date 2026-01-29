function [result,rate] = SDADL_classification4(P,W,Ytr,Yte,train_gnd,test_gnd)
%%%%%% 用 KNN 算法进行分类
%输入参数： 
%         P——解析字典；
%         W——线性分类器
%         Yte——测试样本矩阵；
%         Ytr——训练样本矩阵；
%         train_gnd——训练样本标签；
%         test_gnd -- 测试样本标签；
%输出参数： 
%         classification——输出正确的类别标签；

 Xtr = P*Ytr; % 训练样本的编码系数
 Xtr_label = W * Xtr; % 标签矩阵
 Xtr_label = Xtr_label./repmat(sqrt(sum(Xtr_label.*Xtr_label) + eps),[size(Xtr_label,1) 1]); %标准化
 Xte = P*Yte; % 测试样本的编码系数
 Xte_label = W * Xte; % 标签矩阵
 Xte_label = Xte_label./repmat(sqrt(sum(Xte_label.*Xte_label) + eps),[size(Xte_label,1) 1]); %标准化
 
 Xtr_label = Xtr_label';
 Xte_label = Xte_label';

[result,rate]=NNClassfier(Xte_label,Xtr_label,test_gnd,train_gnd);
end