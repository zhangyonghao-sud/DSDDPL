function [P,W] = ASDPL(Y,Yte,Y_gnd,options)
%%% 输入参数：
%    Y--训练样本 
%    Yte -- 测试样本
%    Y_gnd--训练样本Y的标签 
%%% 输出参数：
%    W --线性分类器
%    P --分析字典

alpha          = options.alpha; 
beta           = options.beta;
lambda         = options.lambda;          
gama           = options.gama;
rho            = options.rho; %分析字典F函数
rho1           = options.rho1;
rate_rho       = options.rate_rho;
lambda1        = options.lambda1;       % 常数 1e-6
iterations     = options.iterations;    % iteration number 用于训练阶段
numofatoms_perclass   = options.numofatoms_perclass; % nunofatoms_perclass 每类分析字典原子数
extendDim      = options.extendDim;      % 用于类标签矩阵的扩展
isExtend       = options.isExtend;       % 如果为1，则扩展类标签矩阵H；为0，则不扩展

%%%%%%%%%% 初始化 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dim   = size(Y,1);   % 求 Y 的行数,即样本向量的维度
NumofClass   = max(Y_gnd);  % 样本类别数
I_Mat   =  eye(Dim,Dim);
YYT     =  Y*Y';  % 用于解析字典的求逆部分
[eigvector, ~] = PCA([Y,Yte]'); % 对训练样本和测试样本联合PCA处理

for i=1:NumofClass
       index_Yi       = find( Y_gnd==i );
       Ni             = length( index_Yi ); % Ni给出Yi中样本的个数
       Yi{i}          = Y(:,index_Yi);   % Yi{i}表示每一类的样本集Yi
       Yi_c{i}        = Y(:,Y_gnd~=i);   % 样本集 Yi 的补集                      
       DataInvMat{i}  = inv(YYT+beta*Yi_c{i}*Yi_c{i}'+rho*I_Mat); 
       Q_Mat{i}       = ones(numofatoms_perclass,Ni); % 判别稀疏编码 Q 的子块
       H_Mat{i}       = ones(1,Ni); % H 的子块
       X{i} = eigvector( :, 1:numofatoms_perclass)'* Yi{i};   % 初始化编码系数 X 的对角块      
end
Q  = blkdiag(Q_Mat{:});  % 判别稀疏编码 Q
H  = blkdiag(H_Mat{:});  % 类标签矩阵 H 
if isExtend % 判断是否扩展
      H = kron(H,ones(extendDim,1));   % H进行扩展
end
%% 初始化编码系数X
X = blkdiag(X{:});
XXT  =  X*X';
%% 初始化分析字典,综合字典P，D
for j=1:NumofClass       
        r_start_X  =  (j-1)*numofatoms_perclass+1; % 按行搜索
        r_end_X    =  j*numofatoms_perclass;
        Xi{j}      =  X(r_start_X:r_end_X,:); % 对应 Xi 部分
end
for k=1:NumofClass
         Pi{k}  = Xi{k}*Y'*DataInvMat{k};
end
P   = cell2mat(Pi'); % 解析字典P的初始化
P   = P./repmat(sqrt(sum(P.*P,2) + eps),[1,size(P,2)]); %解析字典P按行标准化
D   = Y*X'/(lambda1*eye(size(X,1))+X*X');
D   = D./ repmat(sqrt(sum(D.*D)+ eps),[size(D,1) 1]); % 归一化
A   = Q*X'*inv(XXT+lambda1*trace(XXT)*eye(size(XXT))); % 变换矩阵A的初始化,见LC-KSVD
Xcolumn = size(X,2);
Z = eye(Xcolumn,Xcolumn);
%%%%%%%%%%%%%%% 初始化结束(D,P,X,A,Z) %%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% 训练阶段
for i=1:iterations 
    %%%------更新编码系数X------%%%
    Xrow = size(X,1);
    Xcolumn = size(X,2);
    I1 = eye(Xrow,Xrow);
    I2 = eye(Xcolumn,Xcolumn);
    Am = (D'*D)+(alpha*I1)+(gama*(A'*A));
    Bm = lambda*I2-lambda*Z+lambda*Z'+lambda*(Z*Z');
    Cm = D'*Y+alpha*P*Y+gama*A'*Q;
    X= sylvester(Am,Bm,Cm);
    %%%------更新分析字典P------%%%
    for k=1:NumofClass
         %Pi{k} = Xi{k}*Y'*inv(YYT+beta*Yi_c{i}*Yi_c{i}'+rho*I_Mat);
         Pi{k} = Xi{k}*Y'*DataInvMat{k};
    end
    P = cell2mat(Pi'); 
    P = P./repmat(sqrt(sum(P.*P,2) + eps),[1,size(P,2)]);
    %%%------更新综合字典D------%%%
    Imat= eye(size(X,1));
    TempCoef       = X;
    TempData       = Y;
    TempS          = D;
    TempT          = zeros(size(TempS));
    previousD = D;
    Iter=1;ERROR=1;
    while(ERROR>1e-8&&Iter<100)
         TempD   = (rho1*(TempS-TempT)+TempData*TempCoef')/(rho1*Imat+TempCoef*TempCoef');
         TempS   = normcol_lessequal(TempD+TempT);
         TempT   = TempT+TempD-TempS;
         rho1    = rate_rho*rho1;
         ERROR = mean(mean((previousD- TempD).^2));
         previousD = TempD;
         Iter=Iter+1;
    end     
    D   = TempD; 
    %%%------更新转换矩阵Z------%%%
    X_column1 = size(X,2);
    I3 = eye(X_column1);
    Z = (Y'*Y+X'*X+I3)\((Y'*Y)+X'*X);
    %%%------更新转换矩阵A------%%%
    X_column2 = size(A,2);
    I4 = eye(X_column2);
    A = Q*X'*(X*X'+lambda1*I4);
end
W = H*X'/(X*X'+lambda1*trace(XXT)*eye(size(XXT)));
end
    
          
 