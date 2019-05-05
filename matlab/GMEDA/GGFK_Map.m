function [X_new,L_GCA,M] = GGFK_Map(X,indicator,dim,type)
    %Input X = [Xt_1;...,Xt_nt;Xs_1;...;Xs_ns], s=nt+ns
    %X in R^(N,D),D dimension of data, N example numbers
    %indicator(i,:)=[start,end,flag]
    %dim: feature dimension
    %M:GGFK 
    %done
    [M,L_GCA] = GGFK_core(X',indicator,dim,type);
    sq_M = real(M^(0.5));
    X_hat = build_hat(X,indicator);
    X_new = (sq_M * X_hat')';
end

function X_hat = build_hat(X,indicator)
    X_hat = [];
    for i=1:size(indicator,1)
        [sti,edi] = position(i,indicator);
        X_hat = blkdiag(X_hat,X(sti:edi,:));
    end
end

function [M,L_GCA] = GGFK_core(X,indicator,dim,type)
    %done
    eps = 1e-20;
    [P,pca_indicator] = full_pca(X,indicator,dim);
    [U1,U2,theta] = compute_para(P, pca_indicator,dim);
    L_GCA = compute_L(theta,type,dim);
    s = size(pca_inicator,1);
    M = zeros(dim * s, dim * s);
    for i = 1:s
        [sti,edi] = position(i,pca_indicator);
        for j = 1:s
            [stj,edj] = position(j,pca_indicator);
            for k=1:s
                theta_add = theta((i-1)*dim+1:i*dim,k)+theta((j-1)*dim+1:j*dim,k);
                theta_minus = theta((i-1)*dim+1:i*dim,k)-theta((j-1)*dim+1:j*dim,k);
                add_eps = max(eps,theta_add);
                minus_eps = 2.*(double(theta_minus>0)-1).*max(abs(theta_minus),eps);
                lambda1 = diag(sin(theta_add)./2./add_eps+sin(theta_minus)./2./minus_eps);
                lambda2 = diag((cos(theta_add)-1)./2./add_eps-(cos(theta_minus)-1)./2./minus_eps);
                lambda3 = diag((cos(theta_add)-1)./2./add_eps+(cos(theta_minus)-1)./2./minus_eps);
                lambda4 = diag(-sin(theta_add)./2./add_eps+sin(theta_minus)./2./minus_eps);
                M((i-1)*dim+1:i*dim,(j-1)*dim:j*dim) = M((i-1)*dim+1:i*dim,(j-1)*dim:j*dim)... 
                    +[P(:,sti:edi)*U1((i-1)*dim+1:i*dim,(k-1)*dim+1:k*dim),null(P(:,sti:edi)')*U2((i-1)*dim+1:i*dim,(k-1)*dim+1:k*dim)]...
                    *[lambda1,lambda2;lambda3,lambda4]...
                    *[P(:,stj:edj)*U1((j-1)*dim+1:j*dim,(k-1)*dim+1:k*dim),null(P(:,stj:edj)')*U2((j-1)*dim+1:j*dim,(k-1)*dim+1:k*dim)]';
            end
        end
    end

end

function L = compute_L(theta,type,dim)
    s = size(theta,2);
    L = zeros(s,s);
    switch type
        case 'MaxC'
            for i=1:s
                for j=1:s
                    L(i,j) = cos(min(theta((i-1)*dim:i*dim,j)));
                end
            end
        case 'MinC'
            for i=1:s
                for j=1:s
                    L(i,j) = cos(max(theta((i-1)*dim:i*dim,j)));
                end
            end
        case 'Procrusters'
            for i=1:s
                for j=1:s
                    L(i,j) = norm(cos(theta((i-1)*dim:i*dim,j)),2)/sqrt(dim);
                end
            end
        otherwise
            error(['Unsupported type ' type])
    end
end


function [P,pca_indicator] = full_pca(X,indicator,dim)
    %do pca for all datsets in X
    %limit dim = dim
    %done
    P = [];
    pca_indicator = [];
    st = 0;
    ed = 0;
    for i=1:size(indicator,1)
        [sti,edi] = position(i,indicator);
        Pi = pca(X(:,sti:edi)')';
        P = [P,Pi(:,1:dim)];
        st = ed + 1;
        ed = st + size(Pi,2) - 1;
        pca_indicator = [pca_indicator;[st,ed,indicator(i,3)]];
    end
        
end

function [st,ed] = position(i,indicator)
    % offer position for the ith dataset
    % done
    st = indicator(i,1);
    ed = indicator(i,2);
end

function [U1,U2,theta] = compute_para(X,indicator,dim)
    %compute U1ik,U2ik,thetaik
    %
    s = size(indicator,1);
    U1 = zeros(dim*s,dim*s); 
    U2 = zeros(dim*s,dim*s);
    theta = zeros(dim*s,s);
    for i=1:s
        [sti,edi] = position(i,indicator);
        for j=1:s
            [stj,edj] = position(j,indicator);
            [U1((i-1)*dim+1:i*dim,(j-1)*dim+1:j*dim),U2((i-1)*dim+1:i*dim,(j-1)*dim+1:j*dim),theta((i-1)*dim+1:i*dim,j)] = GeoFlow(X(:,sti:edi),X(:,stj:edj));
        end
    end
            
end

function [U1,U2,theta] = GeoFlow(Ps, Pt)
    % compute the geodesic flow for Ps to Pt
    % Ps and Pt are pca componets
    % Pt only contain the firt dim colomn of the real one
    % done
    Q = [Ps,null(Ps')];
    N = size(Q,2);
    dim = size(Pt,2);
    QPt = Q' * Pt;
    [U1,U2,~,Gam,~] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
    U2 = -U2;
    theta = real(acos(diag(Gam)));
end