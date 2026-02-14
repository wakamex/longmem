# Error Analysis: 27 Wrong Questions (Baseline K=50 MR=±1)
# Total: 106/133 correct (79.7%), 27 wrong

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| Judge disagreement | 10 | Model answer matches gold, strict judge rejected |
| Retrieval failure ("I don't know") | 6 | Evidence not in retrieved context |
| Hallucination on abstention | 1 | Model answered when it should have said "I don't know" |
| Wrong count/reasoning | 11 | Model found evidence but computed wrong answer |

## 1. Judge Disagreement (10 questions)

The model gave the CORRECT answer but the LongMemEval strict judge
rejected it. These are false negatives — our actual accuracy may be
closer to 87% (106+10=116/133).

### 00ca467f
**Q:** How many doctor's appointments did I go to in March?
**Gold:** 2
**Model:** Based on the observations, you went to two doctor's appointments in March:

1. **March 3rd** with your primary care physician, Dr. Smith [62].
2.
**Judge:** No.

### 28dc39ac
**Q:** How many hours have I spent playing games in total?
**Gold:** 140 hours
**Model:** Based on the observations, you have spent a total of **140 hours** playing games: 70 hours on *Assassin's Creed Odyssey* [7], 5 hours on *Hyper
**Judge:** No.

### 4f54b7c9
**Q:** How many antique items did I inherit or acquire from my family members?
**Gold:** 5
**Model:** Based on the observations, you inherited or acquired 5 items from your family members: an antique tea set (from cousin Rachel), a vintage typewriter (from your dad), a vintage diamond necklace (from your
**Judge:** No.

### 5a7937c8
**Q:** How many days did I spend participating in faith-related activities in December?
**Gold:** 3 days.
**Model:** Based on the observations, you participated in faith-related activities on 3 days in December: a church food drive on December 10th, a Bible study on December 17th,
**Judge:** No.

### 60036106
**Q:** What was the total number of people reached by my Facebook ad campaign and Instagram influencer collaboration?
**Gold:** 12,000
**Model:** I don't know the mileage of the Hana Highway. The total number of people reached was 12,000 (2,000 from the Facebook ad campaign and 10
**Judge:** No

### 88432d0a
**Q:** How many times did I bake something in the past two weeks?
**Gold:** 4
**Model:** Based on the observations, you baked four times in the past two weeks:

1.  **Sourdough bread** on Tuesday (Observations [17], [19], [
**Judge:** No

### a9f6b44c
**Q:** How many bikes did I service or plan to service in March?
**Gold:** 2
**Model:** Based on the observations, you serviced or planned to service two bikes in March:

1.  **Road bike**: You cleaned and lubricated the chain on March 2nd [38]
**Judge:** No.

### bf659f65
**Q:** How many music albums or EPs have I purchased or downloaded?
**Gold:** 3
**Model:** Based on the observations, you have downloaded or purchased three items: the album "Happier Than Ever" by Billie Eilish (downloaded on Spotify), the EP "Midnight Sky" by
**Judge:** No.

### d23cf73b
**Q:** How many different cuisines have I learned to cook or tried out in the past few months?
**Gold:** 4
**Model:** Based on the observations, you have learned to cook or tried out 4 different cuisines:

1. **Ethiopian**: You tried a new Ethiopian restaurant [23] and learned to make misir
**Judge:** No

### gpt4_15e38248
**Q:** How many pieces of furniture did I buy, assemble, sell, or fix in the past few months?
**Gold:** 4
**Model:** Based on the observations, you have bought, assembled, or fixed 4 pieces of furniture in the past few months:

1.  **Coffee table**: Bought from West Elm about three weeks ago
**Judge:** No.

## 2. Retrieval Failures — "I don't know" (6 questions)

The model said "I don't know" because the relevant evidence was
not in the top-50 retrieved messages. These are pure retrieval failures.

### 37f165cf
**Q:** What was the page count of the two novels I finished in January and March?
**Gold:** 856
**Model:** I don't know.

### 60036106
**Q:** What was the total number of people reached by my Facebook ad campaign and Instagram influencer collaboration?
**Gold:** 12,000
**Model:** I don't know the mileage of the Hana Highway. The total number of people reached was 12,000 (2,000 from the Facebook ad campaign and 10

### 60bf93ed
**Q:** How many days did it take for my laptop backpack to arrive after I bought it?
**Gold:** 5 days. 6 days (including the last day) is also acceptable.
**Model:** I don't know.

### 7405e8b1
**Q:** Did I receive a higher percentage discount on my first order from HelloFresh, compared to my first UberEats order?
**Gold:** Yes.
**Model:** I don't know.

### 80ec1f4f_abs
**Q:** How many different museums or galleries did I visit in December?
**Gold:** 0. You did not mention visitng any museum in December
**Model:** I don't know.

### gpt4_731e37d7
**Q:** How much total money did I spend on attending workshops in the last four months?
**Gold:** $720
**Model:** I don't know. While the observations mention a $500 fee for a digital marketing workshop in March [46] and a $200 fee for a writing workshop

## 3. Hallucination on Abstention (1 question)

### a96c20ee_abs
**Q:** At which university did I present a poster for my undergrad course research project?
**Gold:** The information provided is not enough. You did not mention presenting a poster for your undergrad course research project.
**Model:** Harvard University.

## 4. Wrong Count/Reasoning (10 questions)

Model found some evidence but computed the wrong number or missed items.
These could be retrieval gaps (missing some evidence) or reasoning errors.

### 09ba9854
**Q:** How much will I save by taking the train from the airport to my hotel instead of a taxi?
**Gold:** $50
**Model:** Based on the observations, a taxi from Narita Airport to Shinjuku costs approximately **$140–$180** (¥15,000–¥

### 0a995998
**Q:** How many items of clothing do I need to pick up or return from a store?
**Gold:** 3
**Model:** You need to pick up or return two items: a navy blue blazer from the dry cleaners and a pair of boots from Zara.

### 1a8a66a6
**Q:** How many magazine subscriptions do I currently have?
**Gold:** 2
**Model:** Based on the observations, you currently have 3 magazine subscriptions: **The New Yorker** (subscribed in February [7, 87]), **Architectural Digest** [23], and the

### 6d550036
**Q:** How many projects have I led or am currently leading?
**Gold:** 2
**Model:** Based on the observations, you have led or are currently leading at least five projects:

1.  A **Marketing Research class project** where you led the data analysis team [8].
2

### c4a1ceb8
**Q:** How many different types of citrus fruits have I used in my cocktail recipes?
**Gold:** 3
**Model:** Based on the observations, you have used 4 types of citrus fruits in your cocktail recipes: orange, lemon, lime, and grapefruit.

### d851d5ba
**Q:** How much money did I raise for charity in total?
**Gold:** $3,750
**Model:** Based on the observations, you raised a total of $8,750 for charity:
*   $1,000 for a local children's hospital (charity bake sale) [11, 17]
*   $250 for a local food bank ("Run for Hunger") [23]
*   $500 for the American Cancer Society (fitness challenge) [48]
*   $2,000 for a local animal shelter [62, 68]
*   $5,000 for a local music education program (benefit concert) [74]

### e3038f8c
**Q:** How many rare items do I have in total?
**Gold:** 99
**Model:** Based on the observations, you have a total of 96 rare items: 57 rare records, 25 rare coins, 12 rare figurines, one antique vase, and

### gpt4_2f8be40d
**Q:** How many weddings have I attended in this year?
**Gold:** I attended three weddings. The couples were Rachel and Mike, Emily and Sarah, and Jen and Tom.
**Model:** 19, 21].
        3.  Sister's wedding [61].
        4.  Friend Jen's wedding [72].

    *   Does

### gpt4_372c3eed
**Q:** How many years in total did I spend in formal education from high school to the completion of my Bachelor's degree?
**Gold:** 10 years
**Model:** You spent a total of 8 years in formal education: 4 years in high school (2010 to 2014) and 4 years to complete your Bachelor's degree.

### gpt4_7fce9456
**Q:** How many properties did I view before making an offer on the townhouse in the Brookside neighborhood?
**Gold:** I viewed four properties before making an offer on the townhouse in the Brookside neighborhood. The reasons I didn't make an offer on them were: the kitchen of the bungalow needed serious renovation, the property in Cedar Creek was out of my budget, the noise from the highway was a deal-breaker for the 1-bedroom condo, and my offer on the 2-bedroom condo was rejected due to a higher bid.
**Model:** Based on the observations, you viewed 5 properties before making an offer on the Brookside townhouse:

1. A 3-bedroom bungalow in Oakwood on January 22nd [

### gpt4_d12ceb0e
**Q:** What is the average age of me, my parents, and my grandparents?
**Gold:** 59.6
**Model:** Based on the observations, the ages are:
*   **User:** 32 (Observation [58])
*   **Mom:** 55 (Observation [37])
