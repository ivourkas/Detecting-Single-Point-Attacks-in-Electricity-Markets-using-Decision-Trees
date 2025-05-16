function [Aeq,beq,lb,ub] = lnzdOPF_fe(Y,P,Pg)

% P(<0): loads
% Pg(>0): generator capacities

% buses: 0,1,2,...,n
% x=[P0,P1,...,Pn,d0,d1,...,dn,P0.1,P0.2,P1.2,...,Pn-1.n,P1.0,...,Pn.n-1]'
% i.e. [bus active power, bus angle, active flows (vertical scan triu(Y))]' -> DC OPF
% For the Case of decoupled OPF we have DC OPF outputs plus [bus reactive power, bus voltage magnitude, reactive flows] 
	
% Aeq: equality constraints matrix
% Aeq=[bus P; flows P; offline units P; slack bus angle]

n=length(Y);
lenP=length(P);
s_flows=length(nonzeros(triu(Y,1)));
counter=1;
lb=[zeros(lenP,1);-pi*ones(n,1);-1000*ones(2*s_flows,1)];
ub=[Pg';pi*ones(n,1);1000*ones(2*s_flows,1)];
Aeq=[];

    for i=1:1:n
        for j=1:1:i-1
            if (imag(Y(j,i))~=0)
                Aeq(lenP+counter,lenP+j)=-imag(Y(j,i));
                Aeq(lenP+counter,lenP+i)=imag(Y(j,i));
                Aeq(lenP+s_flows+counter,lenP+j)=imag(Y(j,i));
                Aeq(lenP+s_flows+counter,lenP+i)=-imag(Y(j,i));
                Aeq(lenP+counter,lenP+n+counter)=1;
                Aeq(lenP+s_flows+counter,lenP+n+s_flows+counter)=1;
                                                    %flow eqs
                Aeq(j,lenP+n+counter)=-1;
                Aeq(j,j)=1;
                Aeq(i,lenP+n+s_flows+counter)=-1;
                Aeq(i,i)=1;
                                                    %flows summed to buses
                counter=counter+1;
            end
        end
    end
    beq=[-P';zeros(size(Aeq,1)-length(P),1)];
    
    for i=1:1:lenP
        if Pg(i)==-100000                           %offline units
            Aeq(size(Aeq,1)+1,:)=[zeros(1,(i-1)),1,zeros(1,size(Aeq,2)-i)];
            beq(size(beq,1)+1)=0;
        end
    end
  
    beq(size(beq,1)+1)=0;                           
    Aeq(size(Aeq,1)+1,:)=[zeros(1,lenP),1,zeros(1,size(Aeq,2)-lenP-1)];
                                                    %slack bus
end