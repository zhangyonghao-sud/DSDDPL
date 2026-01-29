function [ D_Mat ] = UpdateD(  Coef, Data, D_Mat, beta )

rho      = 1;
rate_rho = 1.2;
Imat = eye(size(Coef,1));
TempCoef   = Coef;
TempData   = Data;
TempS      = D_Mat;
TempT      = zeros(size(TempS));
previousD  = D_Mat;
Iter  = 1;
ERROR = 1;
while(ERROR>1e-8&&Iter<100)
      TempD   = (rho*(TempS-TempT) + TempData*TempCoef')/(rho*Imat + beta*Imat + TempCoef*TempCoef');
      TempS   = TempD+TempT;
      TempS   = TempS./ repmat(max(1,sqrt(sum(TempS.*TempS)+ eps)),[size(TempS,1) 1]); % 列向量的2范数平方小于等于1的约束
      TempT   = TempT+TempD-TempS;
      rho     = rate_rho*rho;
      ERROR   = mean(mean((previousD- TempD).^2));
      previousD = TempD;
      Iter    = Iter+1;
end     
D_Mat  = TempD;
