
def calc_epsilon(x_true, x_pred):
    return x_true - x_pred


def calc_epsilon_aj(Aj, epsilon_j):
    return np.sum(np.dot(Aj[:, :, i].T, epsilon_j[:, i]) for i in range(Aj.shape[3]))


def calc_epsilon_a(indices, A, epsilon):
    m = n_viewpoints(indices)

    n_pose_params = size(A, 2)
    epsilon_a = np.empty((n_pose_params, m))

    for j in range(m):
        I = points_by_viewpoint(indices, j)

        Aj = view(A, :, :, I)  # [A_ij for i in I]
        epsilon_j = view(epsilon, :, I)  # [epsilon_ij for i in I]

        epsilon_a[:, j] = calc_epsilon_aj(Aj, epsilon_j)

    return epsilon_a


def calc_epsilon_bi(Bi, epsilon_i):
    return np.sum(Bi[:, :, j].T * epsilon_i[:, j] for j in range(Bi.shape[3]))


def calc_epsilon_b(indices, B, epsilon):
    n = n_points(indices)

    n_point_params = B.shape[2]
    epsilon_b = np.empty((n_point_params, n))

    for i in range(n):
        J = viewpoints_by_point(indices, i)

        Bi = B[:, :, J]  # [Bij for j in J]
        epsilon_i = epsilon[:, J]  # [epsilon_ij for j in J]

        epsilon_b[:, i] = calc_epsilon_bi(Bi, epsilon_i)

    return epsilon_b


def calc_XtX(XS):
    return sum(np.dot(X.T, X) for X in eachslice(XS, dims = 3))

def calc_Uj(Aj):
    return calc_XtX(Aj)

def calc_Vi(Bi):
    return calc_XtX(Bi)


def calc_U(indices, A):
    n_pose_params = A.shape[2]

    m = n_viewpoints(indices)

    U = Array{Float64}(undef, n_pose_params, n_pose_params, m)

    for j in 1:m
        # Aj = [Aij for i in points_by_viewpoint(j)]
        Aj = A[:, :, points_by_viewpoint(indices, j)]
        U[:, :, j] = calc_Uj(Aj)
    return U


def calc_V_inv(indices, B):
    n_point_params = B.shape[2]

    n = n_points(indices)

    V_inv = np.empty((n_point_params, n_point_params, n))

    for i in 1:n
        # Bi = [Bij for j in viewpoints_by_point(i)]
        Bi = B[:, :, viewpoints_by_point(indices, i)]
        Vi = calc_Vi(Bi)
        V_inv[:, :, i] = inv(Vi)

    return V_inv


def calc_W(indices, A, B):
    assert(A.shape[3] == B.shape[3])
    N = A.shape[3]

    n_pose_params, n_point_params = size(A, 2), size(B, 2)

    W = np.empty((n_pose_params, n_point_params, N))

    for index in range(N):
        W[:, :, index] = np.dot(A[:, :, index].T, B[:, :, index])

    return W


def calc_Y(indices, W, V_inv):
    Y = np.copy(W)

    for i in range(n_points(indices)):
        Vi_inv = view(V_inv, :, :, i)

        # Yi = [Yij for j in viewpoints_by_point(i)]
        # note that Yi is still a copy of Wi
        Yi = Y[:, :, viewpoints_by_point(indices, i)]

        for Yij in eachslice(Yi, dims = 3)
            Yij[:] = Yij * Vi_inv  # overwrite  Yij = Wi * inv(Vi)

    return Y


def calc_YjWk(Yj, Wk):
    return np.sum(np.dot(Yj[:, :, i], Wk[:, :, i].T) for i in range(Yj.shape[3]))


def calc_S(indices, U, Y, W):
    assert(Y.shape == W.shape)

    def block(index):
        return slice(n_pose_params * (index-1) + 1, n_pose_params * index)

    n_pose_params = U.shape[1]
    m = n_viewpoints(indices)

    S = np.empty((n_pose_params * m, n_pose_params * m))

    for j, k in itertools.product(range(m), range(m))
        ret = shared_point_indices(indices, j, k)

        if isnothing(ret)
            z = zeros(Float64, n_pose_params, n_pose_params)
            S[block(j), block(k)] = z
            continue

        indices_j, indices_k = ret

        Yj = Y[:, :, indices_j]
        Wk = W[:, :, indices_k]

        S[block(j), block(k)] = -calc_YjWk(Yj, Wk)

        if j == k
            S[block(j), block(j)] += U[:, :, j]
    return S


def calc_e(indices, Y, epsilon_a, epsilon_b):
    N = Y.shape[3]
    n_point_params = Y.shape[1]

    Y_epsilon_b = np.empty((n_point_params, N))

    for i in range(n_points(indices)):
        epsilon_bi = view(epsilon_b, :, i)

        for index in viewpoints_by_point(indices, i)
            Y_epsilon_b[:, index] = view(Y, :, :, index) * epsilon_bi

    e = np.copy(epsilon_a)
    for j in range(n_viewpoints(indices)):
        sub = Y_epsilon_b[:, points_by_viewpoint(indices, j)]
        e[:, j] -= np.sum(sub, dims = 2)

    return e


def calc_delta_a(S, e):
    delta_a = np.linalg.solve(S, vec(e))
    n_pose_params, n_viewpoints = e.shape
    reshape(delta_a, n_pose_params, n_viewpoints)


def calc_delta_b(indices, V_inv, W, epsilon_b, delta_a):
    N = W.shape[3]
    n_point_params = epsilon_b.shape[1]

    W_delta_a = np.empty((n_point_params, N))

    # precalculate W_ij * delta_a_j
    for j in range(n_viewpoints(indices)):
        delta_aj = delta_a[:, j]
        # Wi = [W_ij for j in J]
        for index in points_by_viewpoint(indices, j):
            W_delta_a[:, index] = np.dot(W[:, :, index].T, delta_aj)

    n = n_points(indices)
    delta_b = np.empty((n_point_params, n))
    for i in range(n):
        # sum 'W_ij * delta_a_j' over 'j'
        sub = W_delta_a[:, viewpoints_by_point(indices, i)]
        delta_b[:, i] = np.dot(V_inv[:, :, i], epsilon_b[:, i] - np.sum(sub, dims = 2))
    return delta_b


def check_params(indices, x_true, x_pred, A, B):
    assert(A.shape[3] == B.shape[3] == x_true.shape[2] == x_pred.shape[2])
    assert(A.shape[1] == B.shape[1] == 2)

    n_visible_keypoints = x_true.shape[2]
    n_pose_params = A.shape[2]
    n_point_params = B.shape[2]

    n_rows = 2 * n_visible_keypoints
    n_cols_a = n_pose_params * n_viewpoints(indices)
    n_cols_b = n_point_params * n_points(indices)

    # J' * J cannot be invertible if n_rows(J) < n_cols(J)
    if n_rows < n_cols_a + n_cols_b:
        raise ValueError("n_rows(J) must be greater than n_cols(J)")


def sba(indices, x_true, x_pred, A, B):
    check_params(indices, x_true, x_pred, A, B)

    U = calc_U(indices, A)
    V_inv = calc_V_inv(indices, B)
    W = calc_W(indices, A, B)
    Y = calc_Y(indices, W, V_inv)
    S = calc_S(indices, U, Y, W)
    epsilon = calc_epsilon(x_true, x_pred)
    epsilon_a = calc_epsilon_a(indices, A, epsilon)
    epsilon_b = calc_epsilon_b(indices, B, epsilon)
    e = calc_e(indices, Y, epsilon_a, epsilon_b)
    delta_a = calc_delta_a(S, e)
    delta_b = calc_delta_b(indices, V_inv, W, epsilon_b, delta_a)

    return delta_a, delta_b
