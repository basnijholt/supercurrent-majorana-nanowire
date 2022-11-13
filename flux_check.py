hopping_phases = {}


def peierls(func, ind, a, z_interface, c=constants):
    def with_phase(s1, s2, p):
        hop = func(s1, s2, p).astype("complex128")
        y1, y2, z1 = s1.tag[1], s2.tag[1], s1.tag[2]
        z0 = (z_interface[y1] + z_interface[y2]) / 2 - z1

        hopping_phases[(s1.tag, s2.tag)] = z0
        hopping_phases[(s2.tag, s1.tag)] = -z0

        if p.orbital:
            phase = [0, z0 * p.B_x * a**2, 0][ind]
            phi = np.exp(-1j * 1e-18 * c.eV / c.hbar * phase)
            if hop.shape[0] == 2:
                hop *= phi
            elif hop.shape[0] == 4:
                hop *= np.array([phi, phi.conj(), phi, phi.conj()], dtype="complex128")
        return hop

    return with_phase


pos = np.array([i.pos for i in syst.sites])
tags = [i.tag for i in syst.sites]
neighbors_list = [
    list(syst.graph.out_neighbors(i)) for i in range(syst.graph.num_nodes)
]


def find_neighbor(num_node, direction, neighbors_list, tags):
    """Finds the neighbor of a given site.

    Parameters
    ----------
    num_node : int
        Number of the site of which neigbors are to be found.
    direction : list or tuple
        [1, 0, 0] corresponds to x-direction
        [0, -1, 0] corresponds to negative y-direction
    neighbors_list : list
        List with all neighbors per site.
    tags : list
        Tags of each site."""
    neighbors = neighbors_list[num_node]
    tag_from = tags[num_node]
    for neighbor in neighbors:
        tag_to = tags[neighbor]
        delta = tag_to - tag_from
        if delta == direction:
            return neighbor
    return None


# Check algorithm
def find_loops(neighbors_list, tags):
    paths = []
    for node, _ in enumerate(tags):
        #   * (node_4) --- * (node_3)
        #   |              |
        #   * (node_1) --- * (node_2)
        try:
            node_1 = node
            node_2 = find_neighbor(node_1, [0, 1, 0], neighbors_list, tags)
            node_3 = find_neighbor(node_2, [0, 0, 1], neighbors_list, tags)
            node_4 = find_neighbor(node_3, [0, -1, 0], neighbors_list, tags)
            if node_4 is not None:
                paths.append([node_1, node_2, node_3, node_4])
        except TypeError:
            paths.append(None)
    return paths


paths = find_loops(neighbors_list, tags)
fluxes = []
errors = []
for path in paths:
    if path is not None:
        node_1, node_2, node_3, node_4 = path
        first_hop = hopping_phases.get((tags[node_1], tags[node_2]), 0)
        second_hop = hopping_phases.get((tags[node_3], tags[node_4]), 0)
        flux = first_hop + second_hop
        fluxes.append(flux)
        if flux != -1 and flux != 0 and flux != -0.5:
            errors.append(tags[node_1])
            errors.append(tags[node_2])
            errors.append(tags[node_3])
            errors.append(tags[node_4])
print(
    "Max and min flux through a unit cell: min={}, max={}.".format(
        min(fluxes), max(fluxes)
    )
)
print(f"Fluxes: {set(fluxes)}")
pos = np.array([i.tag for i in syst.sites])
try:
    im = hv.Scatter(pos[np.where(pos[:, 0] == 0)][:, 1:]) * hv.Scatter(
        np.array(errors)[:, 1:]
    )
except:
    im = hv.Scatter(pos[np.where(pos[:, 0] == 0)][:, 1:])
im
