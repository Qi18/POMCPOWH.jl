struct POWHNodeBelief{S,A,O,P}
    model::P
    a::A # may be needed in push_weighted! and since a is constant for a node, we store it
    o::O
    dist::CategoricalVector{Tuple{S,Float64}}

    POWHNodeBelief{S,A,O,P}(m,a,o,d) where {S,A,O,P} = new(m,a,o,d)
    function POWHNodeBelief{S, A, O, P}(m::P, s::S, a::A, sp::S, o::O, r) where {S, A, O, P}
        cv = CategoricalVector{Tuple{S,Float64}}((sp, convert(Float64, r)),
                                                 obs_weight(m, s, a, sp, o))
        new(m, a, o, cv)
    end
end

function POWHNodeBelief(model::POMDP{S,A,O}, s::S, a::A, sp::S, o::O, r) where {S,A,O}
    POWNodeBelief{S,A,O,typeof(model)}(model, s, a, sp, o, r)
end

rand(rng::AbstractRNG, b::POWHNodeBelief) = rand(rng, b.dist)
state_mean(b::POWHNodeBelief) = first_mean(b.dist)
POMDPs.currentobs(b::POWHNodeBelief) = b.o
POMDPs.history(b::POWHNodeBelief) = tuple((a=b.a, o=b.o))


struct POWHNodeFilter end

belief_type(::Type{POWHNodeFilter}, ::Type{P}) where {P<:POMDP} = POWHNodeBelief{statetype(P), actiontype(P), obstype(P), P}

init_node_sr_belief(::POWHNodeFilter, p::POMDP, s, a, sp, o, r) = POWHNodeBelief(p, s, a, sp, o, r)

function push_weighted!(b::POWHNodeBelief, ::POWHNodeFilter, s, sp, r)
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, (sp, convert(Float64, r)), w)
end

struct StateBelief{SRB<:POWHNodeBelief}
    sr_belief::SRB
end

rand(rng::AbstractRNG, b::StateBelief) = first(rand(rng, b.sr_belief))
mean(b::StateBelief) = state_mean(b.sr_belief)
POMDPs.currentobs(b::StateBelief) = currentobs(b.sr_belief)
POMDPs.history(b::StateBelief) = history(b.sr_belief)
