"""
	TVdenoise!(output::Vector{Float64}, y::Vector{Float64}, λ::Float64)

	Solve a Total Variation problem with noisy input signal `input` and
	regularization tension `λ`.

	Straight implementation of L. Condat's algorithm, from 	'A Direct Algorithm
	for	1D Total Variation Denoising'
	(http://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/publis/Condat-fast_TV-SPL-2013.pdf)

"""
function TVdenoise!(output::Vector{Float64}, y::Vector{Float64}, λ::Float64)
	n = length(y)

	k = k0 = kmin = kplus = 1
	vmin = y[1] - λ
	vmax = y[1] + λ
	umin = -λ
	umax = λ

	while true
		## 1.
		if k == n
			output[n] = vmin + umin
			return
		end

		if y[k+1] + umin < vmin-λ
			output[k0:kmin] .= vmin
			k = k0 = kplus = kmin = kmin+1
			vmin = y[k]
			vmax = y[k]+2λ
			umin = λ
			umax = -λ
		elseif y[k+1] + umax > vmax+λ
			output[k0:kplus] .= vmax
			k = k0 = kplus = kmin = kplus+1
			vmin = y[k]-2λ
			vmax = y[k]
			umin = λ
			umax = -λ
		else
			k = k+1
			umin += y[k]-vmin
			umax += y[k]-vmax
			if umin ≥ λ
				vmin += (umin-λ) / (k-k0+1)
				umin = λ
				kmin = k
			end
			if umax ≤ -λ
				vmax += (umax+λ) / (k-k0+1)
				umax = -λ
				kplus = k
			end
		end

		k < n && continue

		if umin < 0
			output[k0:kmin] .= vmin
			k = k0 = kmin = kmin+1
			vmin = y[k]
			umin = λ
			umax = y[k] + λ - vmax
			continue
		elseif umax > 0
			output[k0:kplus] .= vmax
			k = k0 = kplus = kplus+1
			vmax = y[k]
			umax = -λ
			umin = y[k] - λ - vmin
			continue
		else
			output[k0:end] .= vmin + umin/(k-k0+1)
			return
		end
	end
end