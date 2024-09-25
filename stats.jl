### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ baf2ba8c-37d4-4b51-9a97-03be2b908a0e
using DataFrames, XLSX, DataFramesMeta, Distributions, CategoricalArrays

# ╔═╡ 24a360a4-b55c-4d83-ba80-ddb0677bc6ba
using Statistics, LinearAlgebra

# ╔═╡ 7ebd1b9e-e8d3-4e9f-abd8-a5d0bf0eb593
using AlgebraOfGraphics, CairoMakie

# ╔═╡ 7a909ab3-2e89-4538-ae70-fc2fb97e81ba
md"""
# Connectives-modality notebook
"""

# ╔═╡ c2245974-dfe5-4f86-9bc2-9f92ad4f73fa
# using Profile, ProfileSVG, Cthulhu

# ╔═╡ 15992835-c4ba-4773-832f-5a5584a05c6d
begin
	myfont = "Source Han Sans JP"
	set_aog_theme!()
	update_theme!(
		font = myfont, 
		fontsize = 28,
		Legend = (
	        labelfont=myfont,
	        titlefont=myfont,
		), Axis = (
			labelfont=myfont,
			xticklabelfont=myfont,
			yticklabelfont=myfont,
	        titlefont=myfont,
		))
end

# ╔═╡ 9f94d512-d70d-440b-82a5-489fc3f77916
md"""
## Load data
"""

# ╔═╡ 029684d9-570c-44ed-9305-b3c473653c38
searchdir(path, key) = filter(x->occursin(key, x), readdir(expanduser(path), join=true))

# ╔═╡ 737352e4-5ff2-412e-8397-beffe3f47aa1
filename = expanduser("人科社B-共起データ-2022-07-17.xlsx")

# ╔═╡ 8fe67c27-4a51-4deb-a553-cf32c76a8f40
learners_filename = expanduser("学習者-共起データ-2022-07-17.xlsx")

# ╔═╡ b65763c2-5647-4e23-a653-8ea6f53c94a0
df = vcat(
	DataFrame(XLSX.readtable(filename, "前文込共起統計")...),
	DataFrame(XLSX.readtable(learners_filename, "前文込共起統計")...)
)

# ╔═╡ ef9a1778-35a1-44e1-b363-88763b02b879
"""Reformats connective expression strings, simplifying them.

termpp("くわしくは（詳しくは）") => "詳しくは"
"""
function termpp(s)
	m = match(r"\（(.+)\）", s)
	if m != nothing
		m.captures[1]
	else
		s
	end
end

# ╔═╡ 209edde8-f300-4597-8d29-e71467cce99b
connectives_df = DataFrame(XLSX.readtable(filename, "接続表現定義")...)

# ╔═╡ 547859c8-ff62-4457-a3f7-e9500a4a97a0
connectives_df.conjunction = termpp.(connectives_df.conjunction)

# ╔═╡ ccd64390-8f58-4873-ad33-15ef299eef0b
connective2function = Dict(Pair.(connectives_df.conjunction, connectives_df.kinou))

# ╔═╡ 7e2bc8d7-ec0a-4f4e-a4d8-cf8b1cdcf4de
length(connective2function)

# ╔═╡ f5cf88a9-f4ac-4c96-bdf1-4f1b04dba04c
transform!(df, :頻度 => ByRow(Int) => :頻度)

# ╔═╡ 0e3ec065-b872-4c7a-bd9f-18e883ab2764
# Replace 'missing' cells with the empty set symbol ∅.
for c ∈ eachcol(df)
	replace!(c, missing => "∅")
end

# ╔═╡ d3bd546a-b8ce-4358-9ab3-f234821dad7f
levels(df.ジャンル)

# ╔═╡ bfa1e03a-beae-43c7-8f8b-eeaae39f4021
begin
	df.ジャンル = categorical(df.ジャンル, levels=[
		"科学技術論文",
		"人文社会学論文",
		"社会科学専門書",
		"BCCWJ*",
		"ベトナム貿易大卒論",
		"Hinoki作文実験",
		"Natane",
		"JCK作文コーパス",
		"学生の日本語意見文",
		"日本語学習者作文コーパス",
	])
	df.前文接続表現 = categorical(termpp.(df.前文接続表現))
	df.前文文末モダリティ= categorical(df.前文文末モダリティ)
	df.接続表現 = categorical(termpp.(df.接続表現))
	df.文末モダリティ = categorical(df.文末モダリティ)
	df.PPM = Float64.(df.PPM)
end

# ╔═╡ 8c2d3a0d-4a08-4eff-b340-58021e355348
df

# ╔═╡ bcb9f003-eaf6-4561-9ac5-0c1de946e980
modality_expressions = sort(unique(m for m in df.文末モダリティ if m != "∅"))

# ╔═╡ be079170-229d-4307-af09-2157d8fa3b37
length(modality_expressions)

# ╔═╡ fd1016fd-4747-42d6-b01c-d907c0714f0d
md"""
## Collocation computations (PMI, dice)
"""

# ╔═╡ 183eebab-8b26-459f-8925-3a3f76a6a2d5
begin
	genres = unique(df[!, :ジャンル])
	terms = [:前文接続表現, :前文文末モダリティ, :接続表現, :文末モダリティ]
	collocation_permutations = [
		[:接続表現, :文末モダリティ],
		[:前文文末モダリティ, :接続表現],
		[:前文文末モダリティ, :接続表現, :文末モダリティ],
		[:前文接続表現, :前文文末モダリティ, :接続表現, :文末モダリティ],
		[:前文接続表現, :接続表現],
	]
end

# ╔═╡ 25ec71a6-491b-4e96-8c4a-76128c591047
function genre_unigram_marginals(df, term)
	Dict(genre => @chain g begin
			groupby(term)
			combine(:頻度 => sum => :頻度)
			Dict(Pair.(_[:, term], _.頻度))
		end
		for ((genre,), g) ∈ pairs(groupby(df, :ジャンル)))
end

# ╔═╡ d1f504f9-ff72-4558-b0f5-424c15f0b976
gums = genre_unigram_marginals(df, :接続表現)

# ╔═╡ 7ff1515f-d3d2-427f-b506-ae9f9b426f09
"""
Pre-computes collocation frequency (optionally per-genre if `genre` is set) as dictionary.
"""
function get_term_counts(df, terms; genres=false)
	if genres 
		Dict(term => @chain df begin
				groupby([:ジャンル, term])
				combine(:頻度 => sum => :頻度)
				genre_unigram_marginals(term)
			end
			for term in terms)
	else
		Dict(term => @chain df begin
				groupby(term)
				combine(:頻度 => sum => :頻度)
				Dict(Pair.(_[:, term], _.頻度))
			end
			for term in terms)
	end
end

# ╔═╡ 8672e1a5-224f-41b9-834e-2940eb8cc0cc
tc = get_term_counts(df, [:接続表現, :文末モダリティ]; genres=true)

# ╔═╡ e8771289-3aae-410b-b4ef-72eceb6f7249
Dict([g, get(d, "それというのも", 0)] for (g, d) in tc[:接続表現])

# ╔═╡ 89bda5db-cb3d-4684-adb7-521471b97ca6
"""
Computes the mutual information of an arbitrary n-gram as in Manning & Schutze 5.4.
`counts` is a vector of marginal counts of each variable, `i` is the collocation frequency and `x` is number of n-grams in the corpus. Binary base is used so unit is bits.

Note that we use the total n-gram count for `x` and not the number of sentences, because there can be more than one collocation per sentence, but this may not be well motivated in either case (the current way seems most straighforward).
"""
function pmi_n(i, counts, x)
	n = length(counts)
	# x and counts must be converted to floats to prevent overflows!
	log2(i * Float64(x)^(n - 1)) - log2(reduce(*, Float64.(counts)))
end

# ╔═╡ 1ae10cb4-66d7-49cd-8837-8895c55a5abc
isapprox(pmi_n(20, [42, 20], 14_307_668), 18.38; atol=0.01) # Manning & Schutze, p. 179

# ╔═╡ 71a6d111-33a1-4dec-ab7b-03e69539511c
"""
Computes the Sørensen–Dice coefficient of an arbitrary n-gram (this is probably only valid for n=2, but the code can compute arbitrary n-grams--perhaps return null for n != 2?).
`counts` is a vector of marginal counts of each variable and `i` is the collocation frequency.

DSC(X,Y) = 2|X∩Y|/(|X|+|Y|)
"""
function dice_n(i, counts)
	n = length(counts)
	# counts must be converted to floats to prevent overflows!
	n * i / reduce(+, Float64.(counts))
end

# ╔═╡ 8b0d8a66-b8ab-4f95-ae3b-cb6e38818509
"""
Given an n-gram cooccurrence matrix `df` (DataFrame), computes collocation statistics on the specified collocation terms `collocation_terms` dictionary, which contains collocation counts for all terms and collocation permutations. When `genres` is set, group results by genre. When `genres_set` is set, filters and sums against the set of genres specified. Note that 
Setting `remove_missing` (default=false) to true will remove unmarked (∅) entries from collocation matrix.

Currently we use the global `connective2function` dictionary to add connective function information to the results.
"""
function collocation_stats(d, collocation_terms; genres=false, genres_set=false, remove_missing=false)
	if remove_missing
		d = @chain d begin
			filter(r -> all(r[c] != "∅" for c in collocation_terms), _)
		end
	end
	if genres
		d = @chain d begin
			select(:ジャンル, :頻度, :PPM, collocation_terms...)
			groupby([:ジャンル, collocation_terms...])
			combine(:頻度 => sum => :頻度, :PPM => sum => :PPM)
		end
	elseif (genres_set !== false && !isempty(genres_set))
		# Here we drop PPM as we would need to do a weighted mean as genre sizes differ (you cannot sum PPM counts from different corpora)
		d = @chain d begin
			filter(r -> r.ジャンル ∈ genres_set, _)
			select(:頻度, collocation_terms...)
			groupby(collocation_terms)
			combine(:頻度 => sum => :頻度)
		end
	else
		# Here we also drop PPM as we would need to do a weighted mean as genre sizes differ (you cannot sum PPM counts from different corpora)
		d = @chain d begin
			select(:頻度, collocation_terms...)
			groupby(collocation_terms)
			combine(:頻度 => sum => :頻度)
		end
	end
	N = sum(d.頻度)
	counts = get_term_counts(d, collocation_terms; genres)
	pmi_marginals(r) = begin
		pmi_n(
			r.頻度,
			genres ? 
			[counts[term][r.ジャンル][getproperty(r, term)] for term in collocation_terms] : 
			[counts[term][getproperty(r, term)] for term in collocation_terms],
			N
		)
	end
	dice_marginals(r) = begin
		dice_n(
			r.頻度,
			genres ? 
			[counts[term][r.ジャンル][getproperty(r, term)] for term in collocation_terms] : 
			[counts[term][getproperty(r, term)] for term in collocation_terms]
		)
	end

	pmi = [pmi_marginals(r) for r ∈ eachrow(d)]
	d.PMI = pmi

	dice = [dice_marginals(r) for r ∈ eachrow(d)]
	d.dice = dice

	# Add connective function annotations
	for annotate_cols ∈ [c for c in names(d) if c == "接続表現" || c == "前文接続表現"]
		d[!, String(annotate_cols) * "_機能分類"] = [
			conn ∈ keys(connective2function) ? connective2function[conn] : missing
			for conn in d[!, annotate_cols]
		]
	end
	
	sort(d, :PMI, rev=true)
end

# ╔═╡ 4124c96b-446a-4210-8085-9163569a920d
@chain collocation_stats(df, [:接続表現, :前文接続表現]; genres=false, genres_set=Set(["BCCWJ*", "科学技術論文"])) filter(x -> x.頻度 > 5, _)

# ╔═╡ fc96bd22-6ee9-4b05-b159-06a588c8931e
md"""
### PMI and Sorensen-dice with empty (unmarked) values per genre
"""

# ╔═╡ 1d2a322d-f194-4571-ab4c-13ba22633222
begin
	pmidicegenre = DataFrames.stack(collocation_stats(df, [:接続表現, :文末モダリティ]; genres=true), [:PMI, :dice], [:ジャンル, :接続表現, :文末モダリティ])
	draw(data(pmidicegenre) * visual(BoxPlot, show_notch=true) * mapping(:variable, :value, color = :ジャンル, dodge=:ジャンル) * mapping(col = :variable), facet = (; linkxaxes = :none, linkyaxes = :none), figure = (resolution = (2000, 2000),))
end

# ╔═╡ 240925b0-9cd9-42b0-9184-ce07ff9feaa5
draw(data(collocation_stats(df, [:接続表現, :文末モダリティ]; genres=true)) * mapping(:PMI, :dice; marker = :文末モダリティ, color = :文末モダリティ, row = :ジャンル, text = :接続表現 => verbatim), facet = (; linkxaxes = :none, linkyaxes = :none), figure = (resolution = (2000, 4000),))

# ╔═╡ ad766fbe-b83c-4871-9224-0cf524f23d16
md"""
### PMI and Sorensen-dice without empty (unmarked) values per genre
"""

# ╔═╡ 8feabc2b-95fc-4ad9-bbf4-1f2e304d76be
begin
	pmidicegenre_removed = DataFrames.stack(collocation_stats(df, [:接続表現, :文末モダリティ]; genres=true, remove_missing=true), [:PMI, :dice], [:ジャンル, :接続表現, :文末モダリティ])
	draw(data(pmidicegenre_removed) * visual(BoxPlot, show_notch=true) * mapping(:variable, :value, color = :ジャンル, dodge=:ジャンル) * mapping(col = :variable), facet = (; linkxaxes = :none, linkyaxes = :none), figure = (resolution = (2000, 2000),))
end

# ╔═╡ b3e6a21d-c04a-4eb4-886c-6ff5095f1a5e
draw(data(collocation_stats(df, [:接続表現, :文末モダリティ]; genres=true, remove_missing=true)) * mapping(:PMI, :dice; marker = :文末モダリティ, color = :文末モダリティ, row = :ジャンル, text = :接続表現 => verbatim), facet = (; linkxaxes = :none, linkyaxes = :none), figure = (resolution = (2000, 4000),))

# ╔═╡ 8ebf52aa-dbcd-4852-803c-8dcd6aea5a8f
md"""
## Entropy calculation
"""

# ╔═╡ 607d99d8-0d23-4632-ba14-dd708e842c8d
entropy2(x) = entropy(normalize(x, 1), 2) # == scipy.entropy(x; base=2)

# ╔═╡ a0bf201e-df07-4afb-84da-8e2ad408bb68
"""
Helper function to sort collocations in matrix form ascending by highest value.
Optionally only returns the top k collocates if `top_k` is set > 0.
"""
function matsort(m; top_k=-1)
	sorted_mat = sortslices(m; dims=1, rev=true, by=x->x[2])
	if top_k > 0
		if top_k >= size(m)[1]
			top_k = size(m)[1]
		end
		sorted_mat[1:top_k, :]
	else
		sorted_mat
	end
end

# ╔═╡ e034e53d-4eb2-4d33-bc06-0dde09b36c9c
matsort(["∅" 6727;
        "だ" 140900000;
        "となる" 340555]; top_k=1)

# ╔═╡ ebc8234c-71cb-4f27-a880-70074631d193
"""
Computes the ratio of unmarked collocates in collocate matrix.
"""
function unmarked_ratio(M::Matrix)
	unmarked_row = findfirst(x -> x == "∅", M)[1]
	if isnothing(unmarked_row)
		0.0
	else
		unmarked_freq = M[unmarked_row, end]
		unmarked_freq / sum(M[:, end])
	end
end

# ╔═╡ afc06698-36b8-462d-a110-8a6f0f2a469c
unmarked_ratio(
	["∅" 6727;
     "だ" 1409;
     "となる" 340]
)

# ╔═╡ c0e89921-d997-41bc-98eb-80f33a541952
"""
Helper function for `term_entropy` below.
"""
function term_entropy_1(df, term, genres, grouping_terms, min_freq, remove_missing)
	@chain df begin
	filter(r -> remove_missing ? (r[term] != "∅") : true, _)
	groupby(genres ? [:ジャンル, term, grouping_terms...] : [term, grouping_terms...])
	combine(:頻度 => sum => :頻度, :PPM => sum => :PPM) # FIXME PPM => weighted_mean if summing across corpora
	subset(:頻度 => ByRow(>=(min_freq)))
	groupby(genres ? [:ジャンル, term] : term)
	transform(
		AsTable([grouping_terms..., :頻度]) => (x -> entropy2(x.頻度)) => String(term) * "_H",

		grouping_terms => (x -> length(unique(v for v in x if v != "∅"))) => String(term) * "_共起異なり数"
	)
	groupby(genres ? [:ジャンル, term, Symbol(String(term) * "_H"), Symbol(String(term) * "_共起異なり数")] : [term, Symbol(String(term) * "_H"), Symbol(String(term) * "_共起異なり数")])
	combine(
		:頻度 => sum => :頻度,
		:PPM => sum => :PPM,
		# AsTable([grouping_terms..., :頻度]) => (x -> 
		# unmarked_ratio(hcat(collect(x[p] for p in keys(x))...))) => :無標率,
		AsTable([grouping_terms..., :頻度]) 
		=> (x -> begin
			join([join(r, "+", ": ")
				  for r in eachrow(matsort(hcat(collect(x[p] for p in keys(x) if p != "∅")...)))], ", ")
		end) => String(term) * "_共起"
	)
	sort([term, "頻度"]; rev=true)
	end
end

# ╔═╡ e82a5c82-6f65-4da0-8a47-820354c4934f
"""# Term entropy
Calculates the entropy value of each term in the given terms group.
The entropy is calculated for each term against the rest. If `min_freq` is provided, then only terms having a frequency value greater than or equal to it will be considered.
Missing terms (∅) for each term in `terms` are not ignored when calculating the entropy value but considered a term of their own, unless `remove_missing` is true. 
"""
function term_entropy(df, terms; genres=false, genres_set=false, min_freq=0, remove_missing=true)
	term_groups = [
		[term, [t for t ∈ terms if t != term]]
		for term ∈ terms
	]
	if (genres_set !== false && !isempty(genres_set))
		df = filter(r -> r.ジャンル ∈ genres_set, df)
	end
	if remove_missing # Make sure to only remove non-term in term-groups
		df = filter(r -> all(r[t] != "∅" for t in terms), df)
	end
	entropies = 
		Dict(
			String(term) * "統計" => term_entropy_1(df, term, genres, grouping_terms, min_freq, remove_missing)
			for (term, grouping_terms) ∈ term_groups
		)
	for (k, v) in entropies
		# Add connective function annotations
		for annotate_cols ∈ [c for c in names(v) if c == "接続表現" || c == "前文接続表現"]
			v[!, String(annotate_cols) * "_機能分類"] = [
				conn ∈ keys(connective2function) ? connective2function[conn] : missing
				for conn in v[!, annotate_cols]
			]
		end
		# Shouldn't need to be run. If genres is not set we are dealing with multiple corpora of differing size, so PPM counts are based on different normalization factors. These should not be used in any sum opertations, so we delete them here.
		if !genres && (genres_set !== false)
			select!(v, Not(:PPM))
		end
		h_column_idx = findfirst(c -> occursin("_H", c), names(v))
		sort!(v, [names(v)[isnothing(h_column_idx) ? 1 : h_column_idx], :頻度]; rev=[true, true])
	end
	entropies
end

# ╔═╡ ef7c87a5-8d13-480b-b94f-414cfc4d4888
tf = term_entropy(df, [:接続表現, :文末モダリティ]; genres=false, genres_set=Set(["BCCWJ*", "科学技術論文", "人文社会学論文", "社会学専門書"]), min_freq=0, remove_missing=true)

# ╔═╡ 2d789f5e-cd63-43fe-950a-e04de3f6e398
begin
	academic = Set(["科学技術論文", "人文社会学論文", "社会科学専門書", "BCCWJ*"])
	learner = Set(["Natane", "ベトナム貿易大卒論", "JCK作文コーパス", "Hinoki作文実験", "日本語学習者作文コーパス", "学生の日本語意見文"])
    temp = term_entropy(df, [:接続表現, :文末モダリティ]; genres=true, remove_missing=true)
	temp_learner = term_entropy(df, [:接続表現, :文末モダリティ]; genres=false, genres_set=learner, remove_missing=true)
	
	function sieve(k, v)
		filter!(r -> r.ジャンル ∈ academic, v)
		filter!(r -> r.頻度 > 50, v)
		select!(v, Not(:PPM))
		v = vcat(v, transform(temp_learner[k], :頻度 => (x -> "学習者") => :ジャンル))
		filter!(r -> r.頻度 > 5, v)
		v
	end
	temp = Dict(k => sieve(k, v) for (k, v) in temp)
	
	# for (k, v) in temp
	# 	filter!(r -> r.ジャンル ∈ academic, v)
	# 	filter!(r -> r.頻度 > 50, v)
	# 	select!(v, Not(:PPM))
	# 	v = vcat(v, transform(temp_learner[k], :頻度 => (x -> "学習者") => :ジャンル))
	# 	filter!(r -> r.頻度 > 10, v)
	# end
	temp_c = @chain temp["接続表現統計"] begin
			select(:頻度 => (x -> "接続表現") => :項目, :接続表現 => :表現, :接続表現_H => :H, :頻度, :ジャンル)
        end
	temp_m = @chain temp["文末モダリティ統計"] begin
			select(:頻度 => (x -> "文末モダリティ") => :項目, :文末モダリティ => :表現, :文末モダリティ_H => :H, :頻度, :ジャンル)
        end
	temp_df = vcat(temp_c, temp_m)
	draw(data(temp_df) * visual(BoxPlot, show_notch=true) * mapping(:項目, :H, color = :ジャンル, dodge=:ジャンル), figure = (resolution = (2000, 2000),))
end

# ╔═╡ fda64cfc-7e9e-441b-a6a3-b0a886c87452
tddf = vcat(
	select(tf["接続表現統計"], :頻度 => (x -> "接続表現") => :項目, :接続表現 => :表現, :接続表現_H => :H, :頻度),
	select(tf["文末モダリティ統計"], :頻度 => (x -> "文末モダリティ") => :項目, :文末モダリティ => :表現, :文末モダリティ_H => :H, :頻度))

# ╔═╡ a6c89829-b43e-4f39-b92e-dc84eb46df36
draw(data(tddf) * mapping(:H, color = :項目) * AlgebraOfGraphics.density(), figure = (resolution = (2000, 2000),))

# ╔═╡ 0a8b7258-eff3-4fed-b115-575fda8f18e0
draw(data(tddf) * mapping(:H, :頻度, color = :項目), figure = (resolution = (2000, 2000),))

# ╔═╡ 29e46fae-b1d6-4e54-8b93-cb82f8d97ed2
draw(data(tddf) * visual(BoxPlot, show_notch=true) * mapping(:項目, :H, color = :項目), figure = (resolution = (2000, 2000),))

# ╔═╡ 35d7773e-5d4d-4061-bd87-dc7b3b9dee44
term_entropy(df, [:接続表現, :文末モダリティ])["接続表現統計"]

# ╔═╡ feb48e6a-209c-41c5-8762-388c88d77f79
md"""
## Save results
"""

# ╔═╡ 69d73c88-2682-4ce0-89ab-4b881ea2a017
# Add type conversion to support categorical values as strings in XLSX
XLSX.setdata!(ws::XLSX.Worksheet, ref::XLSX.CellRef, val::CategoricalArrays.CategoricalValue) = XLSX.setdata!(ws, ref, XLSX.CellValue(ws, convert(String, val)))

# ╔═╡ 1bda1e93-ab47-4ec6-ae11-edb2e2d9ca93
XLSX.writetable(
	"ジャンル別統計-2022-07-29.xlsx",
	[(k, collect(DataFrames.eachcol(v)), names(v))
	 for (k, v) ∈ merge(
		 term_entropy(df, [:接続表現, :文末モダリティ], genres=true, remove_missing=false),
		 Dict(
			 join(collocation, "→") => collocation_stats(df, collocation; genres=true, remove_missing=false)
			 for collocation ∈ collocation_permutations
		 )
	 )];
	overwrite=true
)

# ╔═╡ 40579301-01da-44ee-8fed-11a534939021
let
	gset = Set(["BCCWJ*", "科学技術論文", "人文社会学論文", "社会学専門書"])
	XLSX.writetable(
		"B科人社統計-2022-07-29.xlsx",
		[(k, collect(DataFrames.eachcol(v)), names(v))
		 for (k, v) ∈ merge(Dict(
			join(collocation, "→") => collocation_stats(df, collocation; genres_set=gset, remove_missing=false)
			for collocation ∈ collocation_permutations
		 ), term_entropy(df, [:接続表現, :文末モダリティ]; genres_set=gset, remove_missing=false))];
		overwrite=true
	)
end

# ╔═╡ 8257ecf5-15e8-46c4-a3ef-4c170bc1806a
let
	gset = Set(["Natane", "ベトナム貿易大卒論", "JCK作文コーパス", "Hinoki作文実験", "日本語学習者作文コーパス", "学生の日本語意見文"])
	XLSX.writetable(
		"学習者統計-2022-07-29.xlsx",
		[(k, collect(DataFrames.eachcol(v)), names(v))
		 for (k, v) ∈ merge(Dict(
			join(collocation, "→") => collocation_stats(df, collocation; genres_set=gset, remove_missing=false)
			for collocation ∈ collocation_permutations
		 ), term_entropy(df, [:接続表現, :文末モダリティ]; genres_set=gset, remove_missing=false))];
		overwrite=true
	)
end

# ╔═╡ 5954d4dc-93a9-4128-9532-eb7676eccae0


# ╔═╡ 19bf347d-ce56-4de2-a0ea-41c7317cb1f3


# ╔═╡ 8e08ab15-8107-45ff-b6bc-8a558b916447
md"""
## Tables

Below is code that generates the tables used in the paper.
"""

# ╔═╡ 7015a500-f367-4874-a358-539f0352e943
firstchar(s) = first.(s)

# ╔═╡ b3fe9143-f798-460b-8487-1a324aa3cfdb
draw(data(term_entropy(df, [:接続表現, :文末モダリティ])["接続表現統計"]) * visual(BoxPlot, show_notch=false) * mapping(:接続表現_機能分類, :接続表現_H, color = :接続表現_機能分類 => firstchar, group = :接続表現_機能分類), figure = (resolution = (2000, 2000),), axis = (xticklabelrotation = π/2,))

# ╔═╡ df101816-1132-4de6-8847-156d8f3ccca7
# transform!(df_cm, :, :接続表現_機能分類 => firstchar => :機能大分類)

# ╔═╡ a766a505-57bb-476f-bf61-0404f024b335
begin
	genresfile = "ジャンル別統計-2022-07-17.xlsx"
	con = DataFrame(XLSX.readtable(genresfile, "接続表現統計")...)
	mod = DataFrame(XLSX.readtable(genresfile, "文末モダリティ統計")...)
end

# ╔═╡ 9eaf9faf-bb28-4b3b-b6cf-b1e33eb9435a
con

# ╔═╡ 84f2b1d7-fe98-4459-892e-71432bd8fd69
mod

# ╔═╡ 2d33c2a9-446d-4744-af55-4f27e6c63c3f
# all_or_nothing(r, n) = if r.前文文末モダリティ

# ╔═╡ f77660c7-6229-4040-a863-da9554c174b3
@chain df begin
	groupby([:ジャンル, :前文文末モダリティ, :接続表現, :文末モダリティ])
	combine(:頻度 => sum => :頻度)
end

# ╔═╡ 3b74d5d5-2350-44ab-a522-68658cd9cdad
begin
	nativefile = "B科人社統計-2022-07-17.xlsx"
	con_native = DataFrame(XLSX.readtable(nativefile, "接続表現統計")...)
	mod_native = DataFrame(XLSX.readtable(nativefile, "文末モダリティ統計")...)
end

# ╔═╡ 724050a1-c4cd-4f84-b2d1-5d0f1357ffab
function trimcollocates(s; n=3)
	terms = split(s, ", ")
	collocations = [term for term in terms if !occursin("∅", term)]
	numterms = length(collocations)
	numterms = numterms > n ? n : numterms
	join(collocations[1:numterms], ", ")
end

# ╔═╡ f5aa1343-d254-46fb-a635-beb4c5da033b
trimcollocates("∅: 179, のだ: 9, だ: 5, かもしれない: 2")

# ╔═╡ 4e98cad4-8c08-4b7c-84ab-971cf54cc67b
table3 = @chain con_native begin
	filter(r -> r.接続表現_H > 0.0 && r.頻度 > 50, _)
	sort(:接続表現_H)
	select(
		:接続表現,
		:接続表現_H => :H,
		:頻度,
		:接続表現_共起異なり数 => :共起項目数,
		:接続表現_共起 => (x -> trimcollocates.(x)) => :共起頻度トップ３のモダリティ)
	first(5)
end

# ╔═╡ a09a9ebb-c957-4cbc-9951-929881bcdf19
XLSX.writetable("table3.xlsx", table3; overwrite=true)

# ╔═╡ 79e7380f-52da-4611-b4cb-ce6769393f6c
table4 = @chain mod_native begin
	filter(r -> r.文末モダリティ_H > 0.0 && r.頻度 > 50 && r.文末モダリティ != "∅", _)
	sort(:文末モダリティ_H)
	select(
		:文末モダリティ,
		:文末モダリティ_H => :H,
		:頻度,
		:文末モダリティ_共起異なり数 => :項目数,
		:文末モダリティ_共起 => (x -> trimcollocates.(x)) => :共起頻度トップ３の接続表現)
	first(5)
end

# ╔═╡ 945d46bf-d5c6-4318-8c70-2eeedaf38107
XLSX.writetable("table4.xlsx", table4; overwrite=true)

# ╔═╡ 2ddabcd4-d440-4f95-98c1-4ef2dc2ce73e
table5 = hcat([
	@chain con begin
		filter(r -> r.ジャンル == genre && r.接続表現 != "∅", _)
		select(:ジャンル => genre, :接続表現, :PPM => (x->round.(Int, x))=> :PPM)
		sort(:PPM; rev=true)
		first(20)
	end
	for genre in genres
]...; makeunique=true)

# ╔═╡ bd253f02-6e63-4c19-a916-75decf2d9a30
XLSX.writetable("table5.xlsx", table5; overwrite=true)

# ╔═╡ 73de63f7-8fb9-46c3-a0a0-fe3dc9df5b61
table6 = hcat([
	@chain mod begin
		filter(r -> r.ジャンル == genre && r.文末モダリティ != "∅", _)
		select(:ジャンル => genre, :文末モダリティ, :PPM => (x->round.(x; digits=2))=> :PPM)
		sort(:PPM; rev=true)
		first(30)
	end
	for genre in genres
]...; makeunique=true)

# ╔═╡ 10f0f061-21ec-4905-88c1-013ba74e54dc
XLSX.writetable("table6.xlsx", table6; overwrite=true)

# ╔═╡ cf75952f-8ac1-45fa-8302-5bcc6b8878c7
begin
	conmod = DataFrame(XLSX.readtable(genresfile, "接続表現→文末モダリティ")...)
end

# ╔═╡ eaa4bdc5-7f63-4a27-a869-8768e699a8e0
table7 = hcat([
	@chain conmod begin
		filter(r -> r.ジャンル == genre && r.文末モダリティ != "∅" && r.接続表現 != "∅" && r.頻度 >= 10, _)
		sort(:PMI; rev=true)
		select(
			:ジャンル => String(genre), 
			:接続表現, # AsTable() => ByRow(x -> termpp(x.接続表現)) => :接続表現, 
			:文末モダリティ, 
			:頻度, 
			:PMI => (x->round.(x, digits=2)) => :PMI
		)
		first(10)
	end
	for genre in genres[1:6]
]...; makeunique=true)

# ╔═╡ 56f942f0-1492-4ecb-a1f5-24defb57f42c
XLSX.writetable("table7.xlsx", table7; overwrite=true)

# ╔═╡ 9ee5a783-d134-4ba6-b1fe-fde0e607af7c
# Table 8: Nataneなどのコーパスを「学習者作文」としてまとめ卒論と同じカテゴリで２分化する
table8 =
	@chain DataFrame(XLSX.readtable("学習者統計-2022-07-17.xlsx", "接続表現→文末モダリティ")...) begin
		filter(r -> r.文末モダリティ != "∅" && r.接続表現 != "∅" && r.頻度 >= 10, _)
		sort(:PMI; rev=true)
		select(
			:接続表現,
			:文末モダリティ, 
			:頻度, 
			:PMI => (x->round.(x; digits=2)) => :PMI
		)
		first(30)
	end

# ╔═╡ e21dd68e-acf6-4292-ab0b-585699189f4b
XLSX.writetable("table8.xlsx", table8; overwrite=true)

# ╔═╡ 077a3e7e-a666-46ae-87d1-77ff9f6cdbe6
genres

# ╔═╡ 51075444-1590-417c-bc26-a56adaeb15ae
begin
	rowsandsum(d) = (
		コーパス = d.ジャンル[1],
		組数 = ncol(d) == 7 ? "2つ組" : "3つ組",
		項目数 = nrow(d),
		PPM = sum(d.PPM)
	)
	collocations = [[:接続表現, :文末モダリティ], [:前文文末モダリティ, :接続表現, :文末モダリティ]]
	table9 = sort(DataFrame([rowsandsum(@chain collocation_stats(df, collocation; genres=true, remove_missing=true) begin filter(r -> r.ジャンル == genre && all(r[c] != "∅" for c in collocation), _) end)
	for collocation in collocations
	for genre in genres]), ["コーパス"])
	table9
end

# ╔═╡ 91399ab4-0acd-4540-ada6-e0df3b95729b
XLSX.writetable("table9.xlsx", table9; overwrite=true)

# ╔═╡ 9c6fb3a4-c173-4cc1-ab12-e52a1ed31eda
begin # FIXME this is wrong, update with 17 version
	n = @chain DataFrame(XLSX.readtable(expanduser("人科社BDB基本統計.xlsx"), "DB統計")...) begin
		filter(r -> r.集計レベル == "sentences", _)
		combine(["BCCWJ*", "人文社会学論文", "社会科学専門書", "科学技術論文"] => ByRow(+) => :オーセンティック)
		select(:オーセンティック => ByRow(Int) => :オーセンティック)
	end
	
	l = @chain DataFrame(XLSX.readtable(expanduser("学習者DB基本統計.xlsx"), "DB統計")...) begin
		filter(r -> r.集計レベル == "sentences", _)
		select(:全 => ByRow(Int) => :学習者)
	end
	hcat(n, l; makeunique=true)
end

# ╔═╡ 4fdd57e8-8f35-409b-9ae7-bd0f524a117f
let
	gset = Set(["BCCWJ*", "科学技術論文", "人文社会学論文", "社会学専門書"])
	@chain collocation_stats(df, [:文末モダリティ]; genres_set=gset, remove_missing=true) begin
		sort(:頻度; rev=true)
	end
	# XLSX.writetable(
	# 	"B科人社統計-2022-07-13.xlsx",
	# 	[(k, collect(DataFrames.eachcol(v)), names(v))
	# 	 for (k, v) ∈ merge(Dict(
	# 		join(collocation, "→") => collocation_stats(df, collocation; genres_set=gset, remove_missing=true)
	# 		for collocation ∈ collocation_permutations
	# 	 ), term_entropy(df, [:接続表現, :文末モダリティ]; genres_set=gset, remove_missing=true))];
	# 	overwrite=true
	# )
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataFramesMeta = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[compat]
AlgebraOfGraphics = "~0.6.5"
CairoMakie = "~0.7.5"
CategoricalArrays = "~0.10.6"
DataFrames = "~1.3.1"
DataFramesMeta = "~0.10.0"
Distributions = "~0.25.45"
XLSX = "~0.7.8"

[extras]
CPUSummary = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "ba70af8b2f7d62da556f5dd38fb1a2fd55a9f346"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AlgebraOfGraphics]]
deps = ["Colors", "Dates", "Dictionaries", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "KernelDensity", "Loess", "Makie", "PlotUtils", "PooledArrays", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "f47c39e2a2d08a6e221dfc639791c6b5c08a9f7a"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.6.6"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c5aeb516a84459e0318a02507d2261edad97eb75"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["TranscodingStreams"]
git-tree-sha1 = "ef9997b3d5547c48b41c7bd8899e812a917b409d"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.4"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "7b6ad8c35f4bc3bca8eb78127c8b99719506a5fb"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.0"

[[deps.CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "StaticArrays"]
git-tree-sha1 = "4a0de4f5aa2d5d27a1efa293aeabb1a081e46b2b"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.7.5"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a2f1c8c668c8e3cb4cca4e57a8efdb09067bb3fd"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+2"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.Chain]]
git-tree-sha1 = "339237319ef4712e6e5df7758d0bccddf5c237d9"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.10"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[deps.DataFramesMeta]]
deps = ["Chain", "DataFrames", "MacroTools", "OrderedCollections", "Reexport"]
git-tree-sha1 = "ab4768d2cc6ab000cd0cec78e8e1ea6b03c7c3e2"
uuid = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
version = "0.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "1f3b7b0d321641c1f2e519f7aed77f8e1f6cb133"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.29"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "e6c693a0e4394f8fda0e51a5bdf5aef26f8235e9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.111"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EllipsisNotation]]
deps = ["StaticArrayInterface"]
git-tree-sha1 = "3507300d4343e8e4ad080ad24e335274c2e297a9"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.8.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "380053d61bb9064d6aa4a9777413b40429c79901"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.2.0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

[[deps.GeoInterface]]
deps = ["RecipesBase"]
git-tree-sha1 = "6b1a29c757f56e0ae01a35918a2c39260e2c4b98"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "0.5.7"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "169c3dc5acae08835a573a8a3e25c62f689f8b5c"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.6.5"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "437abb322a41d527c197fa800455f79d414f0a3c"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.8"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "f389674c99bfcde17dc57454011aa44d5a260a40"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "46efcea75c890e5d820e670516dc156689851722"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.4"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "63de3b8a5c1f764e4e3a036c7752a632b4f0b8d1"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.16.6"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "c5fb1bfac781db766f9e4aef96adc19a729bc9b2"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.2.1"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test"]
git-tree-sha1 = "70e733037bbf02d691e78f95171a1fa08cdc6332"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.2.1"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e60724fd3beea548353984dc61c943ecddb0e29a"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.3+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "98ca7c29edd6fc79cd74c61accb7010a4e7aee33"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.6.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "b366eb1eb68075745777d80861c6706c33f588ae"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.9"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "Static"]
git-tree-sha1 = "c3668ff1a3e4ddf374fc4f8c25539ce7194dcc39"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.6.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "9022bcaa2fc1d484f1326eaa4db8db543ca8c66d"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "bc7fd5c91041f44636b2c134041f7e5263ce58ae"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.10.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XLSX]]
deps = ["Dates", "EzXML", "Printf", "Tables", "ZipFile"]
git-tree-sha1 = "7fa8618da5c27fdab2ceebdff1da8918c8cd8b5d"
uuid = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"
version = "0.7.10"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "1165b0443d0eca63ac1e32b8c0eb69ed2f4f8127"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╠═7a909ab3-2e89-4538-ae70-fc2fb97e81ba
# ╠═baf2ba8c-37d4-4b51-9a97-03be2b908a0e
# ╠═24a360a4-b55c-4d83-ba80-ddb0677bc6ba
# ╠═7ebd1b9e-e8d3-4e9f-abd8-a5d0bf0eb593
# ╠═c2245974-dfe5-4f86-9bc2-9f92ad4f73fa
# ╠═15992835-c4ba-4773-832f-5a5584a05c6d
# ╠═9f94d512-d70d-440b-82a5-489fc3f77916
# ╠═029684d9-570c-44ed-9305-b3c473653c38
# ╠═737352e4-5ff2-412e-8397-beffe3f47aa1
# ╠═8fe67c27-4a51-4deb-a553-cf32c76a8f40
# ╠═b65763c2-5647-4e23-a653-8ea6f53c94a0
# ╠═ef9a1778-35a1-44e1-b363-88763b02b879
# ╠═209edde8-f300-4597-8d29-e71467cce99b
# ╠═547859c8-ff62-4457-a3f7-e9500a4a97a0
# ╠═ccd64390-8f58-4873-ad33-15ef299eef0b
# ╠═7e2bc8d7-ec0a-4f4e-a4d8-cf8b1cdcf4de
# ╠═f5cf88a9-f4ac-4c96-bdf1-4f1b04dba04c
# ╠═0e3ec065-b872-4c7a-bd9f-18e883ab2764
# ╠═d3bd546a-b8ce-4358-9ab3-f234821dad7f
# ╠═bfa1e03a-beae-43c7-8f8b-eeaae39f4021
# ╠═8c2d3a0d-4a08-4eff-b340-58021e355348
# ╠═bcb9f003-eaf6-4561-9ac5-0c1de946e980
# ╠═be079170-229d-4307-af09-2157d8fa3b37
# ╠═fd1016fd-4747-42d6-b01c-d907c0714f0d
# ╠═183eebab-8b26-459f-8925-3a3f76a6a2d5
# ╠═25ec71a6-491b-4e96-8c4a-76128c591047
# ╠═d1f504f9-ff72-4558-b0f5-424c15f0b976
# ╠═7ff1515f-d3d2-427f-b506-ae9f9b426f09
# ╠═8672e1a5-224f-41b9-834e-2940eb8cc0cc
# ╠═e8771289-3aae-410b-b4ef-72eceb6f7249
# ╠═89bda5db-cb3d-4684-adb7-521471b97ca6
# ╠═1ae10cb4-66d7-49cd-8837-8895c55a5abc
# ╠═71a6d111-33a1-4dec-ab7b-03e69539511c
# ╠═8b0d8a66-b8ab-4f95-ae3b-cb6e38818509
# ╠═4124c96b-446a-4210-8085-9163569a920d
# ╠═fc96bd22-6ee9-4b05-b159-06a588c8931e
# ╠═1d2a322d-f194-4571-ab4c-13ba22633222
# ╠═240925b0-9cd9-42b0-9184-ce07ff9feaa5
# ╠═ad766fbe-b83c-4871-9224-0cf524f23d16
# ╠═8feabc2b-95fc-4ad9-bbf4-1f2e304d76be
# ╠═b3e6a21d-c04a-4eb4-886c-6ff5095f1a5e
# ╠═8ebf52aa-dbcd-4852-803c-8dcd6aea5a8f
# ╠═607d99d8-0d23-4632-ba14-dd708e842c8d
# ╠═a0bf201e-df07-4afb-84da-8e2ad408bb68
# ╠═e034e53d-4eb2-4d33-bc06-0dde09b36c9c
# ╠═ebc8234c-71cb-4f27-a880-70074631d193
# ╠═afc06698-36b8-462d-a110-8a6f0f2a469c
# ╠═c0e89921-d997-41bc-98eb-80f33a541952
# ╠═e82a5c82-6f65-4da0-8a47-820354c4934f
# ╠═ef7c87a5-8d13-480b-b94f-414cfc4d4888
# ╠═2d789f5e-cd63-43fe-950a-e04de3f6e398
# ╠═fda64cfc-7e9e-441b-a6a3-b0a886c87452
# ╠═a6c89829-b43e-4f39-b92e-dc84eb46df36
# ╠═0a8b7258-eff3-4fed-b115-575fda8f18e0
# ╠═29e46fae-b1d6-4e54-8b93-cb82f8d97ed2
# ╠═35d7773e-5d4d-4061-bd87-dc7b3b9dee44
# ╠═b3fe9143-f798-460b-8487-1a324aa3cfdb
# ╠═feb48e6a-209c-41c5-8762-388c88d77f79
# ╠═69d73c88-2682-4ce0-89ab-4b881ea2a017
# ╠═1bda1e93-ab47-4ec6-ae11-edb2e2d9ca93
# ╠═40579301-01da-44ee-8fed-11a534939021
# ╠═8257ecf5-15e8-46c4-a3ef-4c170bc1806a
# ╠═5954d4dc-93a9-4128-9532-eb7676eccae0
# ╠═19bf347d-ce56-4de2-a0ea-41c7317cb1f3
# ╠═8e08ab15-8107-45ff-b6bc-8a558b916447
# ╠═7015a500-f367-4874-a358-539f0352e943
# ╠═df101816-1132-4de6-8847-156d8f3ccca7
# ╠═a766a505-57bb-476f-bf61-0404f024b335
# ╠═9eaf9faf-bb28-4b3b-b6cf-b1e33eb9435a
# ╠═84f2b1d7-fe98-4459-892e-71432bd8fd69
# ╠═2d33c2a9-446d-4744-af55-4f27e6c63c3f
# ╠═f77660c7-6229-4040-a863-da9554c174b3
# ╠═3b74d5d5-2350-44ab-a522-68658cd9cdad
# ╠═724050a1-c4cd-4f84-b2d1-5d0f1357ffab
# ╠═f5aa1343-d254-46fb-a635-beb4c5da033b
# ╠═4e98cad4-8c08-4b7c-84ab-971cf54cc67b
# ╠═a09a9ebb-c957-4cbc-9951-929881bcdf19
# ╠═79e7380f-52da-4611-b4cb-ce6769393f6c
# ╠═945d46bf-d5c6-4318-8c70-2eeedaf38107
# ╠═2ddabcd4-d440-4f95-98c1-4ef2dc2ce73e
# ╠═bd253f02-6e63-4c19-a916-75decf2d9a30
# ╠═73de63f7-8fb9-46c3-a0a0-fe3dc9df5b61
# ╠═10f0f061-21ec-4905-88c1-013ba74e54dc
# ╠═cf75952f-8ac1-45fa-8302-5bcc6b8878c7
# ╠═eaa4bdc5-7f63-4a27-a869-8768e699a8e0
# ╠═56f942f0-1492-4ecb-a1f5-24defb57f42c
# ╠═9ee5a783-d134-4ba6-b1fe-fde0e607af7c
# ╠═e21dd68e-acf6-4292-ab0b-585699189f4b
# ╠═077a3e7e-a666-46ae-87d1-77ff9f6cdbe6
# ╠═51075444-1590-417c-bc26-a56adaeb15ae
# ╠═91399ab4-0acd-4540-ada6-e0df3b95729b
# ╠═9c6fb3a4-c173-4cc1-ab12-e52a1ed31eda
# ╠═4fdd57e8-8f35-409b-9ae7-bd0f524a117f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
