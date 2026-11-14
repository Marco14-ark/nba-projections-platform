#!/bin/bash
# Run on your Mac (Terminal). Prints the HTTP status of every free source the pipeline needs.
# Expected for a working setup: every line ends in 200 (injury PDF may be 404 outside report hours).
H=(-H "Referer: https://www.nba.com/" -H "Origin: https://www.nba.com" -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36")
D=$(TZ=America/New_York date +%Y-%m-%d)
ok=0
for u in "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json" \
         "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json" \
         "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022600030.json" \
         "https://stats.nba.com/stats/scoreboardv2?GameDate=$D&LeagueID=00&DayOffset=0"; do
  c=$(curl -s -o /dev/null -m 20 -w "%{http_code}" "${H[@]}" "$u"); echo "$c  $u"; [ "$c" = "200" ] || ok=1
done
# latest official injury report (walk back 15-minute slots)
for k in $(seq 0 15); do
  t=$(TZ=America/New_York date -v-$((k*15))M +%Y-%m-%d_%I_%M%p 2>/dev/null)
  m=${t:14:2}; t="${t:0:14}$(printf %02d $(( (10#$m/15)*15 )))${t:16}"
  c=$(curl -s -o /dev/null -m 20 -w "%{http_code}" "${H[@]}" "https://ak-static.cms.nba.com/referee/injury/Injury-Report_$t.pdf")
  if [ "$c" = "200" ]; then echo "200  injury report $t"; break; fi
done
exit $ok
