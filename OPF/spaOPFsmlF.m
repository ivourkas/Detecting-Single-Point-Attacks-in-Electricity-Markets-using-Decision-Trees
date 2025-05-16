function spaOPFsmlF(casef,fmx,its)

%% INPUT
%   casef:  case file name
%   fmx:    number between 0.0-0.2 to randomize load levels

fmx=max(0,min(0.2,fmx));    % limits fmx between 0-0.2
optoptns=optimoptions('linprog','Display','none');


%% OUTPUT
fln=strcat(num2str(date),num2str(hour(datetime('now'))));
fln=strcat(fln,num2str(minute(datetime('now'))));
fln=strcat(fln,num2str(round(second(datetime('now')))));


%% LOAD CASE & Y MATRIX

mpc = loadcase(casef);
mpc = ext2int(mpc);         % remove isolated buses and gens;
YBUS=full(makeYbus(mpc));   % make Y matrix of the system


%% DATA FROM LOAD CASE
NB = size(mpc.bus,1);       % number of buses
NG = size(mpc.gen,1);       % number of generators
Pg = zeros(1,NB);           % initialize active power of gens
Pd = mpc.bus(:,3);          % load demand
fln=strcat(fln,'_N',num2str(NB),'_G',num2str(NG),'.csv');

for i = 1:1:NG
    Pg(1, mpc.gen(i,1)) = mpc.gen(i,9);
end
% set Pg with max of each gen

Pg = Pg/mpc.baseMVA;
Pd = -Pd/mpc.baseMVA;
% per unit


%% SEASONAL LOAD MODELING

mx = (1-fmx)*min(sum(Pg),sum(abs(Pd)));
% maximum of the system based on the case gens or loads

Win = mx*[0.625, 0.675, 0.665, 0.655, 0.655, 0.675, 0.75, 0.8, 0.70, 0.65, 0.64, 0.63, 0.62, 0.6, 0.6, 0.64, 0.8, 0.85, 0.875, 0.85, 0.825, 0.775, 0.7, 0.65] ;
Spr = mx*[0.675, 0.7, 0.685, 0.68, 0.675, 0.67, 0.725, 0.775, 0.775, 0.7, 0.675, 0.65, 0.6, 0.575, 0.55, 0.55, 0.575, 0.6, 0.7, 0.775, 0.775, 0.75, 0.725, 0.7];
Sum = mx*[0.75, 0.65, 0.6, 0.55, 0.525, 0.515, 0.5, 0.525, 0.625, 0.775, 0.875, 0.925, 0.92, 0.875, 0.92, 0.92, 0.92, 0.975, 1, 0.975, 0.95, 0.95, 0.92, 0.875];
Aut = mx*[0.625, 0.525, 0.475, 0.45, 0.425, 0.425, 0.45, 0.5, 0.6, 0.7, 0.75, 0.775, 0.77,0.765,0.76, 0.755, 0.745, 0.745, 0.755, 0.8, 0.875, 0.925, 0.915, 0.75];
% 24hr load curve basis

pr2f=[];

for iters=1:1:its
    % random factor
    fc = fmx*rand;
    
    WinMax = (Win + Win*fc); WinMin = (Win - Win*fc);
    SprMax = (Spr + Spr*fc); SprMin = (Spr - Spr*fc);
    SumMax = (Sum + Sum*fc); SumMin = (Sum - Sum*fc);
    AutMax = (Aut + Aut*fc); AutMin = (Aut - Aut*fc);
    
    Winter = (WinMax - WinMin).*rand(1,24) + WinMin;
    Spring = (SprMax - SprMin).*rand(1,24) + SprMin;
    Summer = (SumMax - SumMin).*rand(1,24) + SumMin;
    Autumn = (AutMax - AutMin).*rand(1,24) + AutMin;
    
    WinLoad = Winter'*Pd'/abs(sum(Pd)); SprLoad = Spring'*Pd'/abs(sum(Pd));
    SumLoad = Summer'*Pd'/abs(sum(Pd)); AutLoad = Autumn'*Pd'/abs(sum(Pd));
    
    TotalLoad = sparse([WinLoad ; SprLoad; SumLoad; AutLoad]);
    minhrLd=min(abs((sum(TotalLoad,2))));
    
    
    %% DC OPF set-up
    
    BigAeqTot = [];
    BigbeqTot = [];
    BiglbTot = [];
    BigubTot = [];
    
    for s=1:4
        rl = 24*(s-1)+1;
        % index for each season, each starting at 1,25,49,73
        Load = TotalLoad(rl:rl+23,:);
        BigAeq = []; Bigbeq = []; Biglb = []; Bigub = [];
        
        for i=1:24
            [Aeq,beq,lb,ub] = lnzdOPF_fe(YBUS, Load(i,:),Pg);
            % DC OPF
            
            for g = 1:1:NG
                lb(mpc.gen(g,1)) = mpc.gen(g,10)/mpc.baseMVA;
                % lower gen limit
            end
            
            BigAeq = sparse(blkdiag(BigAeq,Aeq)); Bigbeq = sparse([Bigbeq;beq]);
            Biglb = sparse([Biglb; lb]); Bigub = sparse([Bigub; ub]);
            % expanding to full-day horizon
            
        end
        
        BigAeqTot = sparse([BigAeqTot, BigAeq]); 
        BigbeqTot = sparse([BigbeqTot, Bigbeq]);
        BiglbTot = sparse([BiglbTot, Biglb]); 
        BigubTot = sparse([BigubTot, Bigub]);
        %expanding to full-year horizon
    end
    
    % Number of line flows, active power and voltage angles (variables)
    ElX = 2*length(find(triu(YBUS,1))) + 2*NB;
    
    BigA = zeros(1, 24*ElX);
    Bigb = [0];
    
    % RAMP SET-UP
    
    % different ramp rates based on gen capacity
    %(0.3 for half larger ones and 0.5 for rest)
    
    genlabel = transpose((1:size(mpc.gen(:,9))));
    gen = [genlabel mpc.gen(:,9)]; gen=gen(find(gen(:,2)),:);
    gen = sortrows(gen,2); len = length(mpc.gen(:,9));
    gen = gen(1:round(size(gen,1)/2),1); RampRate = zeros(1,len);
    RampRate(gen) = 0.5; RampRate(find(RampRate==0))=0.3;
    % split gen ramps in 50%*Pn for small & 30%*Pn large
    % !! maybe make the k=round(len/3)? !!
    
    nzg=find(mpc.gen(:,9))'; cntr=1;
    for i = nzg %1:NG
        for j=1:23
            BigA(cntr,(ElX*(j-1)+ mpc.gen(i,1))) = 1;
            BigA(cntr,(ElX*(j)+ mpc.gen(i,1))) = -1;
            Bigb(1,cntr) = RampRate(1,i)*Pg(1,mpc.gen(i,1));
            cntr=cntr+1;
            BigA(cntr,ElX*(j-1)+ mpc.gen(i,1)) = -1;
            BigA(cntr,ElX*(j)+ mpc.gen(i,1)) = 1;
            Bigb(1,cntr) = RampRate(1,i)*Pg(1,mpc.gen(i,1));
            cntr=cntr+1;
            %setting ramp rates inequality funtions using sum of 1 or -1
        end
    end
    cntr=cntr-1; BigA=sparse(BigA);
    
    % COST MATRIX
    
    f = zeros(1,ElX);
    for i = 1:1:NG
        f(1, mpc.gen(i,1)) = mpc.gencost(i,4+mpc.gencost(i,4)-1);
    end
    mxgc=max(f)*1.2; mngc=min(nonzeros(f))*0.5;
    
    Bigf = repmat(f,1,24);
    
    
    %% LINPROG NORMAL OPERATION
    
    BigxTot = []; fvalTot = [];
    allsnvrs=size(BigAeqTot,2)/4;
    for i=1:4
        row=allsnvrs*(i-1) + 1;
        Bigx=[];
        [Bigx, fval] = linprog(Bigf,BigA,Bigb,BigAeqTot(:,row:row+allsnvrs-1),BigbeqTot(:,i),BiglbTot(:,i),BigubTot(:,i),optoptns);
        if isempty(Bigx)
            Bigx = -100000*ones(allsnvrs,1); fval=-100000;
        end
        % flag infeasible
        BigxTot = [BigxTot, Bigx];
        fvalTot = [fvalTot, fval];
        %store results
    end
    
    
    %% ATTACK SET-UP
    % TYPE OF ATTACK & GENERATOR ATTACKED (Random)
    % 1. Ramp Rate 2. Upper Limit 3. Lower Limit 4. Cost of Generation
    attgen =  randsample(nzg,1); typeOfAttack = randi(4);
    switch typeOfAttack
        case 1
            a = 0.01; b = 1; r = (b-a).*rand + a;
            RampRate(1,attgen) =  RampRate(1,attgen)*r;
            gcntr=find(nzg==attgen);
            for j=23*(gcntr-1)+1:23*gcntr
                Bigb(1,j) = RampRate(1,attgen)*Pg(1,mpc.gen(attgen,1));
                Bigb(1,j) = RampRate(1,attgen)*Pg(1,mpc.gen(attgen,1));
            end
        case 2
            a = 0.01; b = 1; r = (b-a).*rand + a;
            for i = 1:1:24
                BigubTot(((i-1)*ElX) +mpc.gen(attgen,1),:) = BigubTot(((i-1)*ElX) +mpc.gen(attgen,1),:)*r;
            end
        case 3
            a = 0.01; b = 1; r = (b-a).*rand + a;
            for i = 1:1:24
                BiglbTot(((i-1)*ElX) +mpc.gen(attgen,1),:) = min(BigubTot(((i-1)*ElX) +mpc.gen(attgen,1),:)*r,0.95*minhrLd);
            end
        case 4
            a = mngc; b = mxgc; r = (b-a).*rand + a;
            for i = 1:1:24
                Bigf(1, ((i-1)*ElX) +mpc.gen(attgen,1)) = r;
            end
        otherwise
            disp('NONE');
    end
    
    
    %% LINPROG ATTACK
    
    BigxTotATT = []; fvalTotATT = [];
    
    for i=1:4
        row = allsnvrs*(i-1) + 1;
        Bigx = [];
        [Bigx, fval] = linprog(Bigf,BigA,Bigb,BigAeqTot(:,row:row+allsnvrs-1),BigbeqTot(:,i),BiglbTot(:,i),BigubTot(:,i),optoptns);
        if isempty(Bigx)
            Bigx = -100000*ones(allsnvrs,1); fval=-100000;
        end
        BigxTotATT = [BigxTotATT, Bigx];
        fvalTotATT = [fvalTotATT, fval];
    end
    nrmF=[BigxTot;fvalTot;1,2,3,4;0,0,0,0;0,0,0,0];
    attF=[BigxTotATT;fvalTotATT;1,2,3,4;typeOfAttack*ones(1,4);attgen*ones(1,4)];
    pr2f=sparse([pr2f,nrmF,attF]);
    %
end

dlmwrite(fln,full(pr2f),'-append');

end
