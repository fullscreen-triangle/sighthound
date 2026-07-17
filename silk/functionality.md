# Silk — Core Functionality (Outline)

Silk is a directional, mobile-first journey companion for the existing rail
network. You give it a destination; standing at any station, it shows you the
next moves toward that destination on the trains that are actually running, and
lets you buy each leg straight from the operator. It owns no trains, holds no
money, and issues no ticket of its own.

> Draft outline — deliberately lean. Bullets to refine together before expanding.

---

## 1. What Silk is (and is not)

- **Is:** an option-mapper. Destination in → "your best next hops toward it, right now" out.
- **Is:** a referral/advisory layer over the operators' *existing* live services.
- **Is not:** a train operator, ticket reseller, or payment holder.
- **Is not:** a fixed-itinerary trip planner (no single locked route/price up front).
- The journey is assembled hop-by-hop by the user's own choices, not sold as a bundle.

## 2. Two anchoring principles

### 2.1 Mobile-first
- Primary context: user standing on/near a platform, phone in hand, deciding *now*.
- Core loop is glanceable: current station → next-hop options → tap → repeat.
- Designed for one-handed, few-second interactions; not desktop planning sessions.
- Location-aware: knows (or is told) which station you're at, surfaces its live departures.

### 2.2 The ticket is a glorified button
- "Buy" in Silk = the *same purchase* you'd make on bahn.de / the operator's app.
- Tapping it **deep-links** into the operator's own checkout for that leg.
- Silk never takes the money — the transaction is always **user ↔ operator**.
- No Silk-issued ticket, no custody, no markup. Silk put the right button in front of you.
- Consequence: no fare regulation / payment-holding burden falls on Silk.

## 3. The core user loop

1. **Set destination** — user enters a final destination (station/place).
2. **Locate** — Silk determines the user's current station (GPS or manual pick).
3. **Show next hops** — Silk lists departing services *from here* that move toward the destination.
4. **Choose** — user picks one (by their own preference: speed, cost, changes, comfort).
5. **Buy (button)** — tap deep-links to the operator's checkout for that leg only.
6. **Board & arrive at next station** — loop returns to step 2 automatically.
7. **Repeat until destination reached** — pressure resolves, journey ends.

- No step commits the user to a downstream leg; every hop is decided fresh on arrival.
- Missing a specific service is a non-event: the loop just shows the next-best hop.

## 4. What counts as a "next hop toward X" (option surface)

- **Directional filter:** a departure qualifies if its next stop reduces distance-to-destination.
- **Default view:** meaningful progress (curated) — hide near-useless nudges / obvious dead-ends.
- **Expand view:** all directional options (fully permissive) — user decides everything.
- **Annotations per option** (where operator data allows): speed class, price, travel time,
  number of onward options at the arrival station.
- Multi-modal by nature: U-Bahn, S-Bahn, RB/RE, IC, ICE all appear as candidate hops.

## 5. Data Silk consumes (read-only)

- Live departures / positions / delays from operator feeds (e.g. DB Timetables / RIS, GTFS-RT).
- A station graph + geography so "distance-to-destination" and the gradient are well-defined.
- Fares/annotations per leg where the operator exposes them (else show what's available).
- Silk stores no booking state of its own; it reflects operator state.

## 6. Explicit non-goals (for this outline)

- No payment processing, escrow, or wallet.
- No Silk-branded ticket, seat reservation, or PNR.
- No guaranteed end-to-end itinerary or price quote up front.
- No attempt to schedule, dispatch, or influence the operators' services.
- No account/loyalty/coverage obligations of an operator.

## 7. Open questions to settle next

- **Gradient metric:** distance-to-destination vs. *expected time-to-arrival* (cost-to-go)?
  (Time-to-go is more robust — avoids stranding on a fast-but-dead-end hop.)
- **Buy hand-off targets:** which operator checkouts do we deep-link to, and how (URL scheme,
  app links, web fallback)?
- **Business model:** referral/lead-based only? (Consistent with "holds nothing.")
- **Offline / poor-signal behaviour** on a platform with no data.
- **Cross-operator annotation gaps:** what to show when a vendor exposes no price/time.
