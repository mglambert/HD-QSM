function out = HDQSM(params)
% Hybrid datafidelity term approach for QSM (HD-QSM)
%
% Input:
%   params - Structure with the following required fields:
%       params.input  -  Local Field Map
%       params.weight  -  Data Fidelity Spatially Variable Weight(recommended = magnitude_data). Not used if not specified
%       params.kernel  -  Dipole Kernel in the Frequency Space
%       params.mask  -  Binary Mask of the ROI
%       params.regweight  -  Regularization Spatially Variable Weight. Not used if not specified
%       params.tol_update  -  Convergence Limit of the Second Stage (Default = 1.0)
%       params.maxOuterIterL1 - Iterations of Fist Stage (Default = 20)
%       params.mu1L1  -  Gradient Consistency Weight of Fist Stage (Default = sqrt(params.mu1L2)
%       params.alphaL1  - Regularization Weight of Fist Stage (Default = sqrt(params.alpha11L2)
%       params.maxOuterIterL2  -  Iterations of Second Stage (Default = 80)
%       params.mu1L2  -  Gradient Consistency Weight of Second Stage 
%       params.alphaL2  -  Regularization Weight of Second Stage
% 
% Output:
%   out - Structure with the following fields:
%       out.x  -  Susceptibily Map
%       out.iterL1  -  Number of Iterations in the Fist Stage
%       out.iterL2  -  Number of Iterations in the Second Stage
%       out.time  -  Total Elapsed Time
%       out.params  -  Input Params
% 
% Example:
%   params = [];
%   params.kernel = kernel;
%   params.weight = mask.*(mag_use/max(mag_use(:)));
%   params.input = mask.*phase_use/phase_scale;
%   params.mask = mask;
%   params.alphaL2 = 10^-4.785;
%   params.mu1L2 = 10 * params.alphaL2;
%   out = HDQSM(params);
%
% Based on the code by Carlos Milovic at https://gitlab.com/cmilovic/FANSI-toolbox


    % Stage 1
    tic;
    N = size(params.input);
    
    if isfield(params,'mu1L1')
        mu = params.mu1L1;
    else
        mu = sqrt(params.mu1L2);
    end
    
    if isfield(params,'alphaL1')
        alpha = params.alphaL1;
    else
        alpha = sqrt(params.alphaL2);
    end

    if isfield(params,'mu2L1')
        mu2 = params.mu2L1;
    else
        mu2 = 1.0;
    end
    
    if isfield(params,'maxOuterIterL1')
        num_iter = params.maxOuterIterL1;
    else
        num_iter = 20;
    end
    
    if isfield(params,'tol_update')
       tol_update  = params.tol_update;
    else
       tol_update = 1e-2;
    end

    if isfield(params,'weight')
        weight = params.weight;
    else
        weight = ones(N);
    end
    
    if isfield(params,'regweight')
        regweight = gradient_calc(params.regweight,1);
        if length(size(regweight)) == 3
            regweight = repmat(regweight,[1,1,1,3]);
        end
    else
        regweight = ones([N 3]);
    end

    z_dx = zeros(N, 'single');
    z_dy = zeros(N, 'single');
    z_dz = zeros(N, 'single');

    s_dx = zeros(N, 'single');
    s_dy = zeros(N, 'single');
    s_dz = zeros(N, 'single');

    x = zeros(N, 'single');
    
    s2 = zeros(N,'single');

    [k1, k2, k3] = ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);

    E1 = 1 - exp(2i .* pi .* k1 / N(1));
    E2 = 1 - exp(2i .* pi .* k2 / N(2));
    E3 = 1 - exp(2i .* pi .* k3 / N(3));

    E1t = conj(E1);
    E2t = conj(E2);
    E3t = conj(E3);

    EE2 = E1t .* E1 + E2t .* E2 + E3t .* E3;

    kernel = params.kernel;
    K2 = abs(kernel).^2;

    Wy = params.input;
            
    z2 = zeros(N,'single');

    for t = 1:num_iter
        tx = E1t .* fftn(z_dx - s_dx);
        ty = E2t .* fftn(z_dy - s_dy);
        tz = E3t .* fftn(z_dz - s_dz);

        x_prev = x;
        Dt_kspace = conj(kernel) .* fftn(z2-s2+Wy);
        x = real(ifftn( (mu * (tx + ty + tz) + mu2*Dt_kspace) ./ (eps + mu2*K2 + mu * EE2) ));

        x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
        fprintf('Iter L1: %4d  -  Solution update L1: %12f\n', t, x_update);

        ll = alpha/mu;
        if t < num_iter
            % update z : gradient varible
            Fx = fftn(x);
            x_dx = real(ifftn(E1 .* Fx));
            x_dy = real(ifftn(E2 .* Fx));
            x_dz = real(ifftn(E3 .* Fx));

            z_dx = max(abs(x_dx + s_dx) - regweight(:,:,:,1)*ll, 0) .* sign(x_dx + s_dx);
            z_dy = max(abs(x_dy + s_dy) - regweight(:,:,:,2)*ll, 0) .* sign(x_dy + s_dy);
            z_dz = max(abs(x_dz + s_dz) - regweight(:,:,:,3)*ll, 0) .* sign(x_dz + s_dz);

            % update s : Lagrange multiplier
            s_dx = s_dx + x_dx - z_dx;
            s_dy = s_dy + x_dy - z_dy;            
            s_dz = s_dz + x_dz - z_dz;  

            z2_inner = real(ifftn(kernel.*Fx))+s2 -Wy;
            z2 = max( abs(z2_inner)-weight/mu2, 0.0 ) .* sign(z2_inner) ;

            s2 = z2_inner - z2;
        end

    end
    out.iterL1 = t;

    % Discrepancy factor
    dphi = ((fftn(params.input)) - (fftn(x) .* kernel));
    dphi = ifftn(dphi);
    dphi = abs(dphi);
    dphi = dphi .*params.mask;
    dphi = dphi/max(dphi(:));

    % Stage 2
    mu = params.mu1L2;
    alpha = params.alphaL2;


    if isfield(params,'maxOuterIterL2')
        num_iter = params.maxOuterIterL2;
    else
        num_iter = 80;
    end


    weight = weight.*weight.*(1-dphi).*(1-dphi); 
    Wy = (weight.*params.input./(weight+mu2));
    
    z_dx = zeros(N, 'single');
    z_dy = zeros(N, 'single');
    z_dz = zeros(N, 'single');

    s_dx = zeros(N, 'single');
    s_dy = zeros(N, 'single');
    s_dz = zeros(N, 'single');

    s2 = zeros(N,'single');

    [k1, k2, k3] = ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);

    E1 = 1 - exp(2i .* pi .* k1 / N(1));
    E2 = 1 - exp(2i .* pi .* k2 / N(2));
    E3 = 1 - exp(2i .* pi .* k3 / N(3));

    E1t = conj(E1);
    E2t = conj(E2);
    E3t = conj(E3);

    EE2 = E1t .* E1 + E2t .* E2 + E3t .* E3;

    for t = 1:num_iter
        ll = alpha/mu;
        % update z : gradient varible
        Fx = fftn(x);
        x_dx = real(ifftn(E1 .* Fx));
        x_dy = real(ifftn(E2 .* Fx));
        x_dz = real(ifftn(E3 .* Fx));

        z_dx = max(abs(x_dx + s_dx) - regweight(:,:,:,1)*ll, 0) .* sign(x_dx + s_dx);
        z_dy = max(abs(x_dy + s_dy) - regweight(:,:,:,2)*ll, 0) .* sign(x_dy + s_dy);
        z_dz = max(abs(x_dz + s_dz) - regweight(:,:,:,3)*ll, 0) .* sign(x_dz + s_dz);

        % update s : Lagrange multiplier
        s_dx = s_dx + x_dx - z_dx;
        s_dy = s_dy + x_dy - z_dy;            
        s_dz = s_dz + x_dz - z_dz;  

        z2 = Wy + mu2*real(ifftn(kernel.*Fx)+s2)./(weight + mu2);

        s2 = s2 + real(ifftn(kernel.*Fx)) - z2;
        % update x : susceptibility estimate
        tx = E1t .* fftn(z_dx - s_dx);
        ty = E2t .* fftn(z_dy - s_dy);
        tz = E3t .* fftn(z_dz - s_dz);

        x_prev = x;
        Dt_kspace = conj(kernel) .* fftn(z2-s2);
        x = real(ifftn( (mu * (tx + ty + tz) + mu2*Dt_kspace) ./ (eps + mu2*K2 + mu * EE2) ));

        x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
        fprintf('Iter L2: %4d  -  Solution update L2: %12f\n', t, x_update);


        if x_update < tol_update
            break
        end
    end
    out.time = toc;toc
    out.x = x;
    out.iterL2 = t;
    out.params = params;
end
